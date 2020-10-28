#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "memtable/inlineskiplist.h"
#include "memory/concurrent_arena.h"
#include "db/dbformat.h"
#include "db/memtable.h"
#include "memory/arena.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/comparator.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/options.h"
#include "rocksdb/slice_transform.h"
#include "rocksdb/write_buffer_manager.h"
#include "test_util/testutil.h"
#include "util/gflags_compat.h"
#include "util/mutexlock.h"
#include "util/stop_watch.h"

using GFLAGS_NAMESPACE::ParseCommandLineFlags;
using GFLAGS_NAMESPACE::RegisterFlagValidator;
using GFLAGS_NAMESPACE::SetUsageMessage;

DEFINE_string(benchmarks, "fillrandom",
              "Comma-separated list of benchmarks to run. Options:\n"
              "\tfillrandom                 -- write N random values\n"
              "\tconcurrentfillonlyrandom   -- one or more write threads write N values concurrently\n"
              "\tfillseq                    -- write N values in sequential order\n"
              "\treadrandom                 -- read N values in random order\n"
              "\treadseq                    -- scan the DB\n"
              "\treadwrite                  -- 1 thread writes while N - 1 threads "
              "do random\n"
              "\t                          reads\n"
              "\tseqreadwrite               -- 1 thread writes while N - 1 threads "
              "do scans\n");

DEFINE_int32(
    num_threads, 1,
    "Number of concurrent threads to run. If the benchmark includes writes,\n"
    "then at most one thread will be a writer");

DEFINE_int32(num_operations, 1000000,
             "Number of operations to do for write and random read benchmarks");

DEFINE_int32(num_scans, 1,
             "Number of times for each thread to scan the inlineskiplist for "
             "sequential read "
             "benchmarks");

DEFINE_int32(item_size, 8, "Number of bytes each item should be");

DEFINE_int64(seed, 0,
             "Seed base for random number generators. "
             "When 0 it is deterministic.");

namespace ROCKSDB_NAMESPACE {

class RandomGenerator {
 private:
  std::string data_;
  unsigned int pos_;

 public:
  RandomGenerator() {
    Random rnd(301);
    auto size = (unsigned)std::max(1048576, FLAGS_item_size);
    data_ = rnd.RandomString(size);
    pos_ = 0;
  }

  Slice Generate(unsigned int len) {
    assert(len <= data_.size());
    if (pos_ + len > data_.size()) {
      pos_ = 0;
    }
    pos_ += len;
    return Slice(data_.data() + pos_ - len, len);
  }
};

enum WriteMode { SEQUENTIAL, RANDOM, UNIQUE_RANDOM };

typedef uint64_t Key;

typedef ROCKSDB_NAMESPACE::MemTable::KeyComparator TestComparator;

#if 0
struct TestComparator {
  typedef Key DecodedType;

  static DecodedType Decode(const char* k) {
    return DecodeFixed64(k);
  }

  static DecodedType decode_key(const char* b) {
    return Decode(b);
  }

  int operator()(const char* a, const char* b) const {
    //return memcmp(a, b, sizeof(DecodedType));

    DecodedType aKey = Decode(a);
    DecodedType bKey = Decode(b);

    if (aKey < bKey) {
      return -1;
    } else if (aKey > bKey) {
      return +1;
    } else {
      return 0;
    }
  }

  int operator()(const char* a, const DecodedType b) const {
    //return memcmp(a, (const char*)&b, sizeof(DecodedType));

    DecodedType aKey = Decode(a);
    if (aKey < b) {
      return -1;
    } else if (aKey > b) {
      return +1;
    } else {
      return 0;
    }
  }
};
#endif

class KeyGenerator {
 public:
  KeyGenerator(Random64* rand, WriteMode mode, uint64_t num)
      : rand_(rand), mode_(mode), num_(num), next_(0) {
    if (mode_ == UNIQUE_RANDOM) {
      // NOTE: if memory consumption of this approach becomes a concern,
      // we can either break it into pieces and only random shuffle a section
      // each time. Alternatively, use a bit map implementation
      // (https://reviews.facebook.net/differential/diff/54627/)
      values_.resize(num_);
      for (uint64_t i = 0; i < num_; ++i) {
        values_[i] = i;
      }
      RandomShuffle(values_.begin(), values_.end(),
                    static_cast<uint32_t>(FLAGS_seed));
    }
  }

  uint64_t Next() {
    switch (mode_) {
      case SEQUENTIAL:
        return next_++;
      case RANDOM:
        return rand_->Next() % num_;
      case UNIQUE_RANDOM:
        return values_[next_++];
    }
    assert(false);
    return std::numeric_limits<uint64_t>::max();
  }

  uint64_t Next(uint64_t index) {
    switch (mode_) {
      case UNIQUE_RANDOM:
        return values_[index];
      default:
        assert(false);
        return std::numeric_limits<uint64_t>::max();
    }
  }


 private:
  Random64* rand_;
  WriteMode mode_;
  const uint64_t num_;
  uint64_t next_;
  std::vector<uint64_t> values_;
};

typedef InlineSkipList<TestComparator> TestInlineSkipList;

class BenchmarkThread {
 public:
  explicit BenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                           uint64_t num_ops, uint64_t* read_hits)
      : table_(table),
        key_gen_(key_gen),
        num_ops_(num_ops),
        read_hits_(read_hits) {}

  virtual void operator()() = 0;
  virtual ~BenchmarkThread() {}

 protected:
  TestInlineSkipList* table_;
  KeyGenerator* key_gen_;
  uint64_t num_ops_;
  uint64_t* read_hits_;
};

class FillBenchmarkThread : public BenchmarkThread {
 public:
  FillBenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                      uint64_t num_ops, uint64_t* read_hits)
      : BenchmarkThread(table, key_gen, num_ops, read_hits), sequence_(0),
        internal_key_size_(sizeof(Key)), 
        encoded_len_(VarintLength(internal_key_size_) + internal_key_size_) {}

  void FillOne() {
    auto key = key_gen_->Next();
    //auto internal_key_size = sizeof(Key);
    //auto encoded_len = VarintLength(internal_key_size) + internal_key_size;
    char* buf = table_->AllocateKey(encoded_len_);
    char* p = EncodeVarint32(buf, internal_key_size_);
    EncodeFixed64(p, key);
    //memcpy(buf, &key, sizeof(Key));
    table_->Insert(buf);
  }

#if 0
  void FillOne() {
    auto key = key_gen_->Next();
    //std::cout << "key: " << key << std::endl;
    auto internal_key_size = 16;
    auto value_size = FLAGS_item_size;
    auto encoded_len = VarintLength(internal_key_size) + internal_key_size + value_size;
    char* buf = table_->AllocateKey(encoded_len);
    KeyHandle handle = static_cast<KeyHandle>(buf);
    char* p = EncodeVarint32(buf, internal_key_size);
    EncodeFixed64(p, key);
    p += 8;
    EncodeFixed64(p, ++sequence_);
    p += 8;
    Slice bytes = generator_.Generate(value_size);
    memcpy(p, bytes.data(), value_size);
    p += value_size;
    assert(p == buf + encoded_len);
    table_->Insert(static_cast<char*>(handle));
    //table_->Insert(buf);
  }
#endif

  void operator()() override {
    std::cout << "num_ops: " << num_ops_ << std::endl;
    for (unsigned int i = 0; i < num_ops_; ++i) {
      FillOne();
    }
  }

 protected:
  uint64_t sequence_;
  uint64_t internal_key_size_;
  size_t encoded_len_;
  RandomGenerator generator_;
};

// here concurrent means more than one write threads are concurrently executed
class ConcurrentFillOnlyBenchmarkThread : public BenchmarkThread {
 public:
  ConcurrentFillOnlyBenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                                uint64_t num_ops, uint64_t* read_hits, uint64_t start_index)
      : BenchmarkThread(table, key_gen, num_ops, read_hits), 
        internal_key_size_(sizeof(Key)), 
        encoded_len_(VarintLength(internal_key_size_) + internal_key_size_), 
        start_index_(start_index) { }

  void FillOne(uint64_t index) {
    auto key = key_gen_->Next(index);
    char* buf = table_->AllocateKey(encoded_len_);
    char* p = EncodeVarint32(buf, internal_key_size_);
    EncodeFixed64(p, key);
    table_->InsertConcurrently(buf);
  }
  
  void operator()() override {
    std::cout << "num_ops: " << num_ops_ << ", start_index: " << start_index_ << std::endl;
    for (unsigned int i = 0; i < num_ops_; ++i) {
      FillOne(start_index_++);
    }
  }

 protected:
  uint64_t internal_key_size_;
  size_t encoded_len_;
  uint64_t start_index_;
};

// here concurrent means one write thread concurrently with one or more read threads
class ConcurrentFillBenchmarkThread : public FillBenchmarkThread {
 public:
  ConcurrentFillBenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                                uint64_t num_ops, uint64_t* read_hits, std::atomic_int* threads_done)
      : FillBenchmarkThread(table, key_gen, num_ops, read_hits) {
    threads_done_ = threads_done;
  }

  void operator()() override {
    // # of read threads will be total threads - write threads (always 1). Loop
    // while all reads complete.
    while ((*threads_done_).load() < (FLAGS_num_threads - 1)) {
      FillOne();
    }
  }

 private:
  std::atomic_int* threads_done_;
};

class ReadBenchmarkThread : public BenchmarkThread {
 public:
  ReadBenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                      uint64_t num_ops, uint64_t* read_hits)
      : BenchmarkThread(table, key_gen, num_ops, read_hits), 
        internal_key_size_(sizeof(Key)), 
        encoded_len_(VarintLength(internal_key_size_) + internal_key_size_) {}

  void ReadOne() {
    auto key = key_gen_->Next();
    //auto internal_key_size = sizeof(Key);
    //auto encoded_len = VarintLength(internal_key_size) + internal_key_size;
    // can not use AllocateKey in multi-thread environment
    //char* buf = table_->AllocateKey(encoded_len_);
    char* buf = new char[encoded_len_];
    char* p = EncodeVarint32(buf, internal_key_size_);
    EncodeFixed64(p, key);
    TestInlineSkipList::Iterator iter(table_);
    iter.Seek(buf);
    if (iter.Valid()) {
      ++*read_hits_;
    }
  }

  void operator()() override {
    for (unsigned int i = 0; i < num_ops_; ++i) {
      ReadOne();
    }
  }

 protected:
  uint64_t internal_key_size_;
  size_t encoded_len_;
};

class SeqReadBenchmarkThread : public BenchmarkThread {
 public:
  SeqReadBenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                         uint64_t num_ops,
                         uint64_t* read_hits)
      : BenchmarkThread(table, key_gen, num_ops, read_hits) {}

  void ReadOneSeq() {
    TestInlineSkipList::Iterator iter(table_);
    for (iter.SeekToFirst(); iter.Valid(); iter.Next()) {
      ;
    }

    ++*read_hits_;
  }

  void operator()() override {
    for (unsigned int i = 0; i < num_ops_; ++i) {
      { ReadOneSeq(); }
    }
  }
};

class ConcurrentReadBenchmarkThread : public ReadBenchmarkThread {
 public:
  ConcurrentReadBenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                                uint64_t num_ops,
                                uint64_t* read_hits,
                                std::atomic_int* threads_done)
      : ReadBenchmarkThread(table, key_gen, num_ops, read_hits) {
    threads_done_ = threads_done;
  }

  void operator()() override {
    for (unsigned int i = 0; i < num_ops_; ++i) {
      ReadOne();
    }

    ++*threads_done_;
  }

 private:
  std::atomic_int* threads_done_;
};

class SeqConcurrentReadBenchmarkThread : public SeqReadBenchmarkThread {
 public:
  SeqConcurrentReadBenchmarkThread(TestInlineSkipList* table, KeyGenerator* key_gen,
                                   uint64_t num_ops, uint64_t* read_hits,
                                   std::atomic_int* threads_done)
      : SeqReadBenchmarkThread(table, key_gen, FLAGS_num_scans, read_hits) {
    threads_done_ = threads_done;
  }

  void operator()() override {
    for (unsigned int i = 0; i < num_ops_; ++i) {
      ReadOneSeq();
    }
    ++*threads_done_;
  }

 private:
  std::atomic_int* threads_done_;
};

class Benchmark {
 public:
  explicit Benchmark(TestInlineSkipList* table, KeyGenerator* key_gen, uint32_t num_threads)
      : table_(table),
        key_gen_(key_gen),
        num_threads_(num_threads) {}

  virtual ~Benchmark() {}
  virtual void Run() {
    std::cout << "Number of threads: " << num_threads_ << std::endl;
    std::vector<port::Thread> threads;
    uint64_t read_hits = 0;
    StopWatchNano timer(Env::Default(), true);
    RunThreads(&threads, true, &read_hits);
    auto elapsed_time = static_cast<double>(timer.ElapsedNanos() / 1000);
    std::cout << "Elapsed time: " << elapsed_time << " us"
              << std::endl;
  }

  virtual void RunThreads(std::vector<port::Thread>* threads,
                          bool write, uint64_t* read_hits) = 0;

 protected:
  TestInlineSkipList* table_;
  KeyGenerator* key_gen_;
  uint64_t num_write_ops_per_thread_ = 0;
  uint64_t num_read_ops_per_thread_ = 0;
  const uint32_t num_threads_;
};

class FillBenchmark : public Benchmark {
 public:
  explicit FillBenchmark(TestInlineSkipList* table, KeyGenerator* key_gen)
      : Benchmark(table, key_gen, 1) {
    num_write_ops_per_thread_ = FLAGS_num_operations;
  }

  void RunThreads(std::vector<port::Thread>* threads, 
                  bool /*write*/, uint64_t* read_hits) override {
    FillBenchmarkThread(table_, key_gen_, num_write_ops_per_thread_, read_hits)();
  }
};

class ConcurrentFillOnlyBenchmark : public Benchmark {
 public:
  explicit ConcurrentFillOnlyBenchmark(TestInlineSkipList* table, KeyGenerator* key_gen)
      : Benchmark(table, key_gen, FLAGS_num_threads) {
    num_write_ops_per_thread_ = (FLAGS_num_operations / FLAGS_num_threads);
  }

  void RunThreads(std::vector<port::Thread>* threads, 
                  bool /*write*/, uint64_t* read_hits) override {
    int32_t remaining = (FLAGS_num_operations % FLAGS_num_threads);
    uint64_t start = 0UL;
    std::cout << "num_write_ops_per_thread: " << num_write_ops_per_thread_ << ", remaining: " << remaining << std::endl;
    for (int i = 0; i < remaining; ++i) {
      std::cout << "start: " << start << std::endl;
      threads->emplace_back(
          ConcurrentFillOnlyBenchmarkThread(table_, key_gen_, num_write_ops_per_thread_ + 1, read_hits, start));
      start += (num_write_ops_per_thread_ + 1);
    }

    for (int i = remaining; i < FLAGS_num_threads; ++i) {
      std::cout << "start: " << start << std::endl;
      threads->emplace_back(
          ConcurrentFillOnlyBenchmarkThread(table_, key_gen_, num_write_ops_per_thread_, read_hits, start));
      start += (num_write_ops_per_thread_);
    }
    
    for (auto& thread : *threads) {
      thread.join();
    }
  }
};

class ReadBenchmark : public Benchmark {
 public:
  explicit ReadBenchmark(TestInlineSkipList* table, KeyGenerator* key_gen)
      : Benchmark(table, key_gen, FLAGS_num_threads) {
    num_read_ops_per_thread_ = FLAGS_num_operations / (FLAGS_num_threads - 1);
  }

  void RunThreads(std::vector<port::Thread>* threads, 
                  bool /*write*/, uint64_t* read_hits) override {
    for (int i = 1; i < FLAGS_num_threads; ++i) {
      threads->emplace_back(
          ReadBenchmarkThread(table_, key_gen_, num_read_ops_per_thread_, read_hits));
    }

    for (auto& thread : *threads) {
      thread.join();
    }

    std::cout << "read hit%: "
              << (100 * static_cast<double>(*read_hits) / FLAGS_num_operations)
              << std::endl;
  }
};

class SeqReadBenchmark : public Benchmark {
 public:
  explicit SeqReadBenchmark(TestInlineSkipList* table)
      : Benchmark(table, nullptr, FLAGS_num_threads) {
    num_read_ops_per_thread_ = FLAGS_num_scans;
  }

  void RunThreads(std::vector<port::Thread>* threads, 
                  bool /*write*/, uint64_t* read_hits) override {
    for (int i = 1; i < FLAGS_num_threads; ++i) {
      threads->emplace_back(SeqReadBenchmarkThread(
          table_, key_gen_, num_read_ops_per_thread_, read_hits));
    }
    
    for (auto& thread : *threads) {
      thread.join();
    }
  }
};

template <class ReadThreadType>
class ReadWriteBenchmark : public Benchmark {
 public:
  explicit ReadWriteBenchmark(TestInlineSkipList* table, KeyGenerator* key_gen)
      : Benchmark(table, key_gen, FLAGS_num_threads) {
    num_read_ops_per_thread_ =
        FLAGS_num_threads <= 1
            ? 0
            : (FLAGS_num_operations / (FLAGS_num_threads - 1));
    num_write_ops_per_thread_ = FLAGS_num_operations;
  }

  void RunThreads(std::vector<port::Thread>* threads, 
                  bool /*write*/, uint64_t* read_hits) override {
    std::atomic_int threads_done;
    threads_done.store(0);
    threads->emplace_back(ConcurrentFillBenchmarkThread(
        table_, key_gen_, num_write_ops_per_thread_, read_hits, &threads_done));

    for (int i = 1; i < FLAGS_num_threads; ++i) {
      threads->emplace_back(
          ReadThreadType(table_, key_gen_, num_read_ops_per_thread_, read_hits, &threads_done));
    }

    for (auto& thread : *threads) {
      thread.join();
    }
  }
};

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  SetUsageMessage(std::string("\nUSAGE:\n") + std::string(argv[0]) +
                  " [OPTIONS]...");
  ParseCommandLineFlags(&argc, &argv, true);

  ROCKSDB_NAMESPACE::Random64 rng(FLAGS_seed);
  ROCKSDB_NAMESPACE::Arena arena;
  ROCKSDB_NAMESPACE::ConcurrentArena allocator;
  ROCKSDB_NAMESPACE::InternalKeyComparator internal_key_comp(
      ROCKSDB_NAMESPACE::BytewiseComparator());
  ROCKSDB_NAMESPACE::TestComparator cmp(internal_key_comp);
  //ROCKSDB_NAMESPACE::TestComparator cmp;
  const char* benchmarks = FLAGS_benchmarks.c_str();
  while (benchmarks != nullptr) {
    std::unique_ptr<ROCKSDB_NAMESPACE::KeyGenerator> key_gen;
    const char* sep = strchr(benchmarks, ',');
    ROCKSDB_NAMESPACE::Slice name;
    if (sep == nullptr) {
      name = benchmarks;
      benchmarks = nullptr;
    } else {
      name = ROCKSDB_NAMESPACE::Slice(benchmarks, sep - benchmarks);
      benchmarks = sep + 1;
    }

    std::unique_ptr<ROCKSDB_NAMESPACE::Benchmark> benchmark;
    if (name == ROCKSDB_NAMESPACE::Slice("fillseq")) {
      ROCKSDB_NAMESPACE::TestInlineSkipList table(cmp, &arena);
      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::SEQUENTIAL, FLAGS_num_operations));
      benchmark.reset(new ROCKSDB_NAMESPACE::FillBenchmark(
          &table, key_gen.get()));
    } else if (name == ROCKSDB_NAMESPACE::Slice("fillrandom")) {
      ROCKSDB_NAMESPACE::TestInlineSkipList table(cmp, &arena);
      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::UNIQUE_RANDOM, FLAGS_num_operations));
      benchmark.reset(new ROCKSDB_NAMESPACE::FillBenchmark(
          &table, key_gen.get()));
    } else if (name == ROCKSDB_NAMESPACE::Slice("concurrentfillonlyrandom")) {
      // MUST use ConcurrentArena instead of Arena, otherwise segment fault will be
      ROCKSDB_NAMESPACE::TestInlineSkipList table(cmp, &allocator);
      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::UNIQUE_RANDOM, FLAGS_num_operations));
      benchmark.reset(new ROCKSDB_NAMESPACE::ConcurrentFillOnlyBenchmark(
          &table, key_gen.get()));
    } else if (name == ROCKSDB_NAMESPACE::Slice("readrandom")) {
      ROCKSDB_NAMESPACE::TestInlineSkipList table(cmp, &arena);
      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::SEQUENTIAL, FLAGS_num_operations));
      // prepare data for InlineSkipList
      benchmark.reset(new ROCKSDB_NAMESPACE::FillBenchmark(
          &table, key_gen.get()));
      std::cout << "Preparing data for read " << std::endl;
      benchmark->Run();

      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::RANDOM, FLAGS_num_operations));
      benchmark.reset(new ROCKSDB_NAMESPACE::ReadBenchmark(
          &table, key_gen.get()));
    } else if (name == ROCKSDB_NAMESPACE::Slice("readseq")) {
      ROCKSDB_NAMESPACE::TestInlineSkipList table(cmp, &arena);
      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::SEQUENTIAL, FLAGS_num_operations));
      // prepare data for InlineSkipList
      benchmark.reset(new ROCKSDB_NAMESPACE::FillBenchmark(
          &table, key_gen.get()));
      std::cout << "Preparing data for read " << std::endl;
      benchmark->Run();

      benchmark.reset(new ROCKSDB_NAMESPACE::SeqReadBenchmark(&table));
    } else if (name == ROCKSDB_NAMESPACE::Slice("readwrite")) {
      ROCKSDB_NAMESPACE::TestInlineSkipList table(cmp, &arena);
      // use UNIQUE_RANDOM to keep consistent with masstree-test
      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::RANDOM, FLAGS_num_operations));
      benchmark.reset(new ROCKSDB_NAMESPACE::ReadWriteBenchmark<
                      ROCKSDB_NAMESPACE::ConcurrentReadBenchmarkThread>(
          &table, key_gen.get()));
    } else if (name == ROCKSDB_NAMESPACE::Slice("seqreadwrite")) {
      ROCKSDB_NAMESPACE::TestInlineSkipList table(cmp, &arena);
      // use UNIQUE_RANDOM to keep consistent with masstree-test
      key_gen.reset(new ROCKSDB_NAMESPACE::KeyGenerator(
          &rng, ROCKSDB_NAMESPACE::UNIQUE_RANDOM, FLAGS_num_operations));
      benchmark.reset(new ROCKSDB_NAMESPACE::ReadWriteBenchmark<
                      ROCKSDB_NAMESPACE::SeqConcurrentReadBenchmarkThread>(
          &table, key_gen.get()));
    } else {
      std::cout << "WARNING: skipping unknown benchmark '" << name.ToString()
                << std::endl;
      continue;
    }

    std::cout << "Running " << name.ToString() << std::endl;
    benchmark->Run();
  }

  return 0;
}

