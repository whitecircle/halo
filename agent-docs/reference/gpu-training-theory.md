# GPU Training Theory

You start a training step, look at `nvidia-smi`, and it shows 100% utilization. That figure reads the same for a step that runs the GPU's arithmetic at its limit and for a step that does almost no arithmetic. Three measurements on one NVIDIA B300 show how far apart two busy GPUs can be.

- The same matrix multiply, `[M, 2880] @ [2880, 2880]` in the 16-bit format bf16 called from PyTorch, runs at 1,231 TFLOP/s with 8,192 rows and at 237 TFLOP/s with 128 rows. The call with 128 rows takes 9.0 µs and the call with 16 rows takes 9.6 µs, so an eighth of the work takes the same time.
- A Python loop of 128 small matrix multiplies takes 1.12 ms. One library call that does the same work takes 0.11 ms.
- `nvidia-smi` shows 100% for a loop of large matrix multiplies and 100% for a loop that only moves memory. It shows 93% for the 128-row multiply of the first measurement, which runs at under a fifth of the rate of the large one.

A GPU has several separate speed limits. A training step hits them one at a time, and each has a different fix. This lecture explains the limits in turn, first on one GPU and then across many, and ends with a method for finding which one your step is hitting.

It is written for someone who trains with a PyTorch loop on one GPU and has never looked inside a GPU or across several. This page is part of the documentation of Halo, a distributed training toolkit, and nothing in its physics is specific to Halo. Lines that describe Halo's own choices are set apart as "In Halo" notes, each with a link to the page that holds the detail. The settings that act on each limit have their own pages under [Optimization](../optimization/README.md) and [Parallelism](../parallelism/README.md).

## How to read the numbers

Every number on this page is one of three kinds, and the text says which.

- A measured number was taken for this lecture on one NVIDIA B300 (device name `NVIDIA B300 SXM6 AC`, driver 580.159.03) with PyTorch 2.11 on CUDA 13, in Halo's image for Blackwell, the GPU generation the B300 belongs to (`halo:blackwell`). The shape is given beside it. A number measured earlier in Halo's own benchmarks is called recorded. It names its hardware, model and shape and links the page that holds the run, and that page says which Halo version it was taken on.
- A specification is a figure from the vendor's documentation.
- A computed number comes with its arithmetic, so you can redo it.

The units are the same throughout:

- A **FLOP** is one floating-point multiply or add. A count of them is written in FLOPs, a rate in FLOP/s, and 1 TFLOP/s is 10¹² FLOP/s.
- Sizes computed from a count are decimal: 1 MB is 10⁶ bytes and 1 GB is 10⁹. Most memory read from PyTorch's counters is in GiB (2³⁰ bytes, 7% larger than a GB). A few recorded peaks are in GB. Every size carries its unit.
- Bandwidth is in bytes per second, GB/s or TB/s. The rate of a link between GPUs is per GPU and per direction.
- Number formats are written in lowercase. An fp32 number takes 4 bytes, a bf16 or fp16 number 2 bytes, an fp8 number 1 byte and an fp4 number half a byte. bf16 is the format the training runs in this lecture store their numbers in, and [Precision](#precision) explains why.
- A shape is written the way PyTorch prints it, `[M, K]`. `@` is a matrix multiply and `×` an ordinary multiplication. "About" marks a rounded value.

The lecture has four parts: [One GPU](#one-gpu), [Many GPUs](#many-gpus), [Around the training step](#around-the-training-step) and [Finding the bottleneck](#finding-the-bottleneck). A [glossary](#glossary-and-notation) at the end lists the terms in the order the lecture introduces them, and then the symbols. The "Check yourself" questions that close some sections have their answers in the last section, [Answers](#answers).

## One GPU

### One training step

Start from the loop you already write. One step loads a batch, runs the forward pass, computes the loss, runs the backward pass and updates the weights. With several GPUs the gradients are also combined across GPUs, which the part on [many GPUs](#many-gpus) covers.

Most of the arithmetic in a transformer is in its linear layers, and when the rows of the batch are short, almost all of it. A linear layer is one matrix multiply, `Y = X @ W`.

A function that runs on the GPU is called a **kernel**, and the kernel that runs a matrix multiply is called a **GEMM**, for general matrix multiply. This lecture writes "matmul" for `a @ b` in model code and "GEMM" for the kernel and its shape. "Kernel" always means a GPU function here. It never means the operating system's kernel or the kernel of a convolution.

- `X` is the input, with shape `[M, K]`. It has `M` rows, one per token, and each row is `K` numbers wide.
- `W` is the weight, `[K, N]`.
- `Y` is the output, `[M, N]`.

PyTorch stores the weight of an `nn.Linear` as `[N, K]` and computes `X @ Wᵀ`, so `layer.weight.shape` prints the transpose of what is written here.

The word "row" has two uses, and the lecture keeps them apart. A row of a GEMM is one token. A row of a batch, with no qualifier, is one sequence, and its length in tokens is the row length.

Each of the `M × N` outputs is a sum of `K` products. The GEMM therefore does `M × K × N` multiplies and about as many adds:

```text
FLOPs of one GEMM = 2 × M × K × N
```

`W` has `K × N` entries, so the forward pass costs 2 FLOPs per weight per token. Write `P` for the number of parameters in the model. If you treat every parameter as a matmul weight, the forward pass costs about `2 × P` FLOPs per token.

The backward pass runs two GEMMs for every linear layer. Call the gradient that arrives from the layer above `dY`. It has the shape of `Y`.

- The first, **dgrad**, computes `dX = dY @ Wᵀ`, a GEMM of shape `[M, N] @ [N, K]`. It passes the gradient on to the layer below.
- The second, **wgrad**, computes `dW = Xᵀ @ dY`, a GEMM of shape `[K, M] @ [M, N]`. It is the gradient of the weight, and it needs the `X` that the forward pass saved.

Each of the two has the same three dimensions as the forward GEMM and costs the same `2 × M × K × N`. The backward pass therefore costs about `4 × P` FLOPs per token. Write `T` for the number of tokens in the step:

```text
FLOPs of one step ≈ 6 × P × T
```

![One linear layer through a step. The forward card, Y = X @ W, saves X. A gap stands for the later layers and the loss. dY comes back from the layer above and forks into wgrad, dW = Xᵀ @ dY, which needs the saved X and passes dW to the update, and dgrad, dX = dY @ Wᵀ, which passes dX to the layer below. Each matmul is 2P, 6P FLOPs per token in all. Below: 2 bytes of weights, 2 of gradients and 4 of moments per parameter, and 268 MB for one hidden state.](../assets/diagrams/step_dataflow.png)

The lower half of the figure counts bytes per parameter. The section on memory explains it.

The count of `6 × P` per token is a first estimate, and you should know what it leaves out:

- A frozen weight needs no wgrad, so it costs `4 × P`.
- Layers below the lowest parameter that requires a gradient need no backward pass, because autograd stops there.
- An embedding lookup is not a GEMM. On Qwen3-8B, an open model of 8.2 billion parameters that this lecture uses for its examples, an input embedding of 151,936 × 4,096 entries is 0.62B of the model's 8.2B parameters, 7.6% (computed).
- Attention does extra arithmetic on its scores, and that part grows with the row length. The paragraph under this list counts it.
- In a mixture-of-experts model each token passes through only a few of the model's expert MLPs, so `6 × P` counts FLOPs that do not run.

The usual count for attention is `12 × S × H` FLOPs per token per layer, where `S` is the row length in tokens and `H` here is the number of heads times the head width. On Qwen3-8B that product is the hidden size, 4,096. Scoring one token against `S` keys costs `2 × S × H`, and mixing the values with those scores costs the same again, which is `4 × S × H` for the forward pass. The backward pass costs twice the forward, as it does for a GEMM, so the total is `12 × S × H`.

On Qwen3-8B, with 0.193B parameters per layer, the linear layers cost 6 × 0.193 × 10⁹ = 1.16 × 10⁹ FLOPs per token per layer and attention costs 49,152 × `S`. The two are equal near `S` = 23,600 tokens, and at 8,000 tokens attention adds a third (computed).

That count charges every key. A causal model lets a token see only the keys before it, and a kernel that skips the rest does about `7 × S × H`: half the forward pass, and a backward pass 2.5 times the forward one because it computes the scores again. By that count the two costs meet near 40,400 tokens (computed). The attention rates measured [later](#measured-by-row-length) use it.

Two steps of the loop are not GEMMs. The loss converts the model's output scores to fp32 before it reduces them to one number. The optimizer update does about 16 FLOPs per parameter by a hand count for AdamW, and its time depends on the bytes it moves and hardly on the arithmetic.

The update costs time per parameter, and so does any reduction of gradients across GPUs. The forward and backward passes cost time per token. So the more tokens a step processes, the smaller the share of the step that goes to the update.

Check yourself:

1. A linear layer has a 4096 × 4096 weight. How many FLOPs does the forward pass take for 8,192 tokens? How many does the whole step take for this layer?
2. The same layer is frozen. How many FLOPs does the step take now?

[Answers](#answers-on-one-training-step) are at the end of the page.

### The chip

A GPU has two rates that scale separately: how fast it does arithmetic, and how fast it moves data between its memory and the chip. Most performance questions come down to which of the two a kernel runs out of first.

The chip is divided into identical units called **streaming multiprocessors**, or SMs. A B300 has 148. Each SM has its own arithmetic units and its own small stores, and all of them run at one clock, the SM clock.

#### The arithmetic rate

Almost all of a transformer's FLOPs run on the **tensor cores**, the units of an SM that are built for matrix multiplication. How fast they go depends on how long you measure. These rates are for `torch.mm` on square bf16 operands, where 8192³ means that `M`, `K` and `N` are all 8,192 (measured on one B300):

| GEMM | From a cool start, samples of about 10 ms | Sustained, samples of 0.5 s back to back |
|---|---|---|
| 4096³ | 1,694 TFLOP/s | not measured |
| 8192³ | 1,793 TFLOP/s | 1,388 TFLOP/s at an SM clock of 1,185 MHz and 1,093 W |
| 12288³ | 1,855 TFLOP/s | not measured |
| 16384³ | 1,887 TFLOP/s | 1,468 TFLOP/s at 1,260 MHz and 1,090 W |

The board's power limit separates the two columns. This board is allowed to draw 1,100 W. A loop of large GEMMs reaches that limit about 40 ms after it starts on a cool GPU. The driver then reports a software power cap and lowers the SM clock from its maximum of 2,032 MHz to about 1,180 to 1,260 MHz, and the rate falls with the clock. The driver never gave a thermal reason in these runs.

A 20 s run at 8192³ settled at 1,384 TFLOP/s, and the figure in [Power and utilization](#power-and-utilization) draws its first second. A single call that takes longer than about 40 ms, such as a 32768³ GEMM at 48 ms, is throttled before it returns.

The limit belongs to the board. 1,100 W is the default and the maximum that this board reports (`power.default_limit` and `power.max_limit` in `nvidia-smi`), and NVIDIA lists the chip itself at up to 1,400 W. So the sustained rate below, and every figure this lecture derives from it, describes this board at its limit and not every B300.

The rest of the lecture calls the two rates by name.

- The **burst GEMM peak** is about 1,800 to 1,900 TFLOP/s. It lasts for about the first 40 ms from a cool start.
- The **sustained GEMM peak** is about 1,400 TFLOP/s: 1,388 at 8192³, and up to 1,472 on the best of the other shapes tried.

A training run lasts for hours, so the sustained GEMM peak is the one a step can approach. Arithmetic on this page uses 1,388. Other runs of the same GEMM on this board read 1,384 to 1,435 TFLOP/s, depending on how long each sample was and on what ran before it. Work that is limited by memory or by the CPU does not reach the power limit and keeps the full 2,032 MHz.

The same GEMM in other number formats shows what the tensor cores contribute (measured on the same B300). In fp16 it runs 5 to 8% below bf16. In fp32, as PyTorch runs an fp32 matmul by default, it runs at 68 to 69.5 TFLOP/s, a twentieth of bf16.

The fp32 matmul is that slow because by default it does not use the tensor cores. It runs on the SM's ordinary arithmetic units, the CUDA cores. Counting one multiply and one add per core per clock cycle, their limit is 148 SMs × 128 CUDA cores × 2 FLOPs × 2.032 GHz = 77 TFLOP/s (computed), and 69.5 is 90% of that.

PyTorch can be told to let fp32 matmuls use the tensor cores in a reduced-precision mode called TF32. The same GEMM then runs at 723 to 776 TFLOP/s, about half of bf16. This lecture writes "full-precision fp32" for the default and "TF32" for the other.

#### Memory and its bandwidth

**HBM**, high-bandwidth memory, is the GPU's main memory. Weights, activations and optimizer state are kept there. A B300 has 288 GB by NVIDIA's count. The driver reports 287.4 GB, which is 267.7 GiB.

HBM bandwidth was measured with bf16 buffers of 1 to 64 GiB, counting the bytes read plus the bytes written (one B300):

| Access pattern | Rate |
|---|---|
| copy from one buffer to another | 6.56 to 6.65 TB/s |
| elementwise read and write | 6.8 TB/s |
| read only | 7.2 to 7.3 TB/s with a plain hand-written kernel, 5.9 to 6.1 TB/s with `torch.sum` |
| write only | 7.5 to 7.6 TB/s with a plain hand-written kernel, 4.0 TB/s with `fill_` |

NVIDIA's specification is 8 TB/s. This lecture calls the copy rate **HBM bandwidth** and writes it as 6.6 TB/s, with 6.65 in arithmetic. The figure counts both the read and the write, so copying 1 GB is 2 GB of traffic.

Two details of the table come back later. Traffic that only reads or only writes runs 10 to 15% faster than a copy. And a kernel can be slower than the memory under it: PyTorch's `fill_` writes at 4.0 TB/s where a plain store loop writes at 7.6.

#### The ratio of the two

Divide one rate by the other. The sustained GEMM peak over HBM bandwidth is 1,388 × 10¹² / 6.65 × 10¹² = 209 FLOPs per byte, or 221 with the 16384³ rate. With the burst GEMM peak it is 270 to 284.

So the tensor cores can do about 210 FLOPs in the time HBM delivers one byte, and that is the figure to remember (arithmetic on this page uses 209). A kernel can reach the sustained GEMM peak only if it does at least that much arithmetic for every byte it moves. Meeting the condition is not enough by itself. NVIDIA's GPU Performance Background User's Guide, in the reading list, names three limiters: memory bandwidth, math bandwidth and latency.

#### Inside the chip

The values in this table are NVIDIA specifications or what the device reports:

| Part | On a B300 | Note |
|---|---|---|
| SMs | 148 | NVIDIA's article on the chip gives 160, and this board's driver reports 148 |
| Tensor cores per SM | 4, fifth generation | |
| CUDA cores per SM | 128 | fp32 and integer arithmetic |
| Registers per SM | 65,536 of 32 bits (256 KiB), at most 255 per thread | the operands an instruction reads |
| Shared memory per SM | 228 KiB (233,472 bytes), of which one block can use 227 KiB | a kernel's own work area; NVIDIA writes these sizes as KB and means 1,024 bytes |
| Tensor memory per SM | 256 KiB (262,144 bytes), as 128 rows × 512 columns of 32-bit cells | new on Blackwell; holds the tensor cores' running result |
| L2 cache | 126.5 MiB, shared by all SMs | sits between HBM and the SMs |
| HBM | 267.7 GiB | |
| Maximum SM clock | 2,032 MHz | |
| Board power limit | 1,100 W | the board idles at 185 W with no process and 237 W with an idle process on it |

Add up the storage on the SMs themselves. Shared memory is 148 × 228 KiB = 34.6 MB, registers are 148 × 256 KiB = 38.8 MB and tensor memory is another 38.8 MB. Together that is 112 MB, 0.04% of HBM (computed). A kernel cannot keep its data on the SMs, so it has to work on a small piece at a time.

The further a store is from the SM, the longer a load from it takes. One thread's random load takes 22 to 24 ns when the data is already in the SM's L1 cache, a small cache that takes its space from the same pool as shared memory. From HBM it takes about 160 ns, which is about 325 cycles of the SM clock. A hit in the L2 cache takes 147 to 156 ns, nearly as long.

That wait is the **memory latency**, the time of one load. It was measured on the B300 with a kernel in which the address of each load comes from the load before it, so no load can start early. In bytes per second the L2 cache is faster than HBM: a copy that stays inside it ran at 8.0 to 9.9 TB/s (measured).

A fast kernel loads a piece into shared memory once, does as much arithmetic on it as it can and writes the result back. A slow kernel reads from HBM, touches each value once and writes it back.

#### Two more numbers called the peak

You will meet two other figures that are called the peak of this GPU. Neither is a rate you can measure.

- The **MFU reference peak** is 2,250 TFLOP/s, an NVIDIA specification and the one Halo's hardware registry holds. NVIDIA lists the 8-GPU HGX B300 at 36 × 10¹⁵ FLOP/s for fp16 and bf16. It quotes that rate for sparse operands and says the rate for ordinary dense ones is half, and an eighth of the dense rate is one GPU. The name comes from MFU, model FLOPs utilization, which divides the FLOP rate of a step by this peak. [MFU](#mfu) defines it in full.
- The **ceiling at the maximum clock** is 2,464 TFLOP/s. It is computed: 148 SMs × 2.032 GHz × 8,192 FLOPs per SM per clock cycle. The 8,192 is a model of what the tensor cores of one SM do per cycle. It is not an NVIDIA figure.

Under the power cap the sustained GEMM does 96 to 97% of the model's 8,192 FLOPs per SM per clock cycle (computed from the rates and clocks in the table of [The arithmetic rate](#the-arithmetic-rate)). A sustained GEMM is therefore limited by the SM clock, and the power limit sets the clock. [MFU](#mfu) uses this to show why an MFU of 1.0 cannot be reached on this board.

By this model no step executes more than 2,464 TFLOP/s in bf16. The fastest GEMM measured here reached 77% of that, and NVIDIA's own figure is lower. If a reported rate is higher than the ceiling, it counts FLOPs that did not run.

> In Halo: the "achieved TFLOPS" column for a sparse mixture-of-experts model is such a figure. It credits parameters that a token does not use, and it reads 3,673 on one gpt-oss-20b row ([Throughput Benchmarks → Maximizing achieved TFLOPS](../optimization/throughput-benchmarks.md#maximizing-achieved-tflops)). That page writes the `6 × P` rule as `6·N`, with `N` for the parameter count.

### How the GPU runs a kernel

A kernel is one function that many threads run at once.

When code launches a kernel, it asks for threads on three levels.

- The hardware runs threads in groups of 32 called **warps**. A warp executes one common instruction at a time, so its threads run in lockstep, and a thread that took the other side of a branch sits that instruction out.
- Warps are grouped into **blocks** of at most 1,024 threads. NVIDIA's other name for a block is CTA, cooperative thread array, and the kernel literature uses it.
- The blocks of one launch form a **grid**.

A block runs on one SM from start to finish, and the blocks of a grid are spread over the SMs.

An SM holds a limited number of blocks at a time. On a B300 an SM holds at most 32 blocks, 64 warps and 2,048 threads. That is NVIDIA's specification for compute capability 10.3, its version number for this GPU's architecture. A B300 therefore has room for 148 × 64 = 9,472 warps.

**Occupancy** is the share of an SM's 64 warp slots that a kernel fills. Three things limit it: the registers each thread needs, the shared memory each block asks for, and the caps of 32 blocks and 64 warps. For a simple elementwise kernel the registers decide. An SM has 65,536 of them, and every thread on it needs its own:

| Registers per thread | Threads that fit | Warps | Occupancy |
|---|---|---|---|
| 32 | 2,048 | 64 | 100% |
| 64 | 1,024 | 32 | 50% |
| 128 | 512 | 16 | 25% |
| 255 | 257 | 8 | 12.5% |

A pipelined GEMM kernel is usually limited by shared memory instead. A block that takes most of an SM's 228 KiB for its own work area is alone on that SM.

You care about occupancy because memory is slow next to arithmetic. A warp that asks for data from HBM has to wait about 160 ns for it. The SM switches to another warp that is ready and comes back when the data has arrived. With many warps on the SM, one of them can run most of the time, so little of the wait is lost. Tensor-core GEMM kernels cover the same wait in another way: a hardware unit copies the next piece of data in the background while the tensor cores multiply the current one.

A grid can have more blocks than the SMs can hold. It then runs in **waves**: the first set of blocks runs, and the rest start as slots become free. A grid with fewer blocks than SMs leaves some SMs with nothing to do.

NVIDIA's rule of thumb, from its GPU performance guide, is that a kernel is likely to have enough parallelism when its grid has at least about 4 times as many blocks as the GPU has SMs, with a few hundred threads in each block. Below that, the guide says, the kernel is limited by latency.

NVIDIA also names the **tail effect**. At the end of a kernel only a few blocks are still running, and the SMs that have finished stand idle until the last block is done. The guide adds that only launches of fewer than about 300 blocks need to be examined for it.

For the bf16 GEMM `[128, 2880] @ [2880, 2880]`, the library that PyTorch calls launches a grid of 92 blocks of 256 threads each, which is 8 warps per block (read from a profiler trace on the B300). A longer GEMM of 1,560 rows at the same width launches 240 blocks, more than the 148 SMs.

![A launch of a matrix multiply as its grid of 92 blocks. Below, 148 SM cells drawn one block to an SM, at most 92 filled and at least 56 empty, then the same cells under a 240-block launch, all filled, with 92 blocks waiting if an SM takes one at a time. Beside them, one block as 8 rows of 32 thread dots, and one SM with 64 warp slots and bars for its three stores.](../assets/diagrams/kernel_blocks_on_sms.png)

The figure draws one block to an SM, and that is an assumption. An SM has warp slots for 8 blocks of 256 threads, and how many blocks of this kernel it runs at the same time depends on the shared memory each block asks for, which was not measured.

> In Halo: the fused AdamW kernel is held to 32 registers per thread so that an SM keeps all 64 warps ([BF16 Optimizer](../optimization/bf16-optimizer.md)).

### The CPU launches and the GPU runs later

When your code calls an operation on a GPU tensor, the CPU puts a kernel into a queue called a **stream** and returns at once. The GPU runs the queued kernels in order while the CPU goes on to launch the next ones. PyTorch's notes on CUDA say it in one line: "By default, GPU operations are asynchronous."

A timer around GPU code therefore measures the time to enqueue the kernel, unless you wait for the GPU. Call `torch.cuda.synchronize()` before you start the clock and again before you stop it, after a few warm-up calls. The same notes warn that "time measurements without synchronizations are not accurate". Try it:

```python
import time
import torch

a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(10):  # warm-up
    a @ b
torch.cuda.synchronize()

start = time.perf_counter()
a @ b
print(f"no synchronize: {(time.perf_counter() - start) * 1e6:.0f} µs")

torch.cuda.synchronize()
start = time.perf_counter()
a @ b
torch.cuda.synchronize()
print(f"synchronized:   {(time.perf_counter() - start) * 1e3:.2f} ms")
```

On the B300 this prints about 30 µs and 0.63 ms (27 to 31 µs and 0.62 to 0.63 ms over five runs). The first number is the CPU's time to launch one GEMM and to allocate its 128 MiB output. In a tight loop the same `a @ b` costs the CPU 10 to 11 µs, and a `torch.mm` that writes into an existing tensor 8 to 9 µs.

The second number is one GEMM from a cool GPU. Run back to back for a second, the same call takes 0.77 to 0.79 ms.

PyTorch can also time on the GPU's own clock. `torch.cuda.Event(enable_timing=True)` records a time at a point in the stream, and the difference between two events is time on the GPU with no CPU in it. The measurements for this lecture were timed with events around runs of calls. One pair of events around a single small kernel costs more than the kernel, about 10 µs here, so time a run of calls.

Two more things belong with this. The environment variable `CUDA_LAUNCH_BLOCKING=1` makes every call wait for its kernel, which lets you trace a CUDA error to the Python line that caused it. And anything that reads a value back to the CPU makes the CPU wait for the whole queue: `.item()`, `.cpu()`, `nonzero`, a Python `if` on a tensor.

#### The launch floor

Launching a kernel costs CPU time, and that time is a floor under every small kernel. The **launch floor** is the CPU time to launch one kernel. These rows are for 10,000 calls each on tiny tensors (measured on one B300):

| Operation | CPU time to launch one | The kernel on the GPU |
|---|---|---|
| `x.add_(1.0)` on 8 elements | 4.80 µs | 0.90 µs |
| `torch.mm` on 8 × 8 bf16 matrices | 7.93 µs | 1.02 µs |

The launch floor is about 5 µs for a trivial operation and about 8 to 9 µs for a `torch.mm`. It moves from one process to another: a 1-element `add_` read 4.93 µs in one process and 6.36 µs in another, and a full-size `torch.mm` read 8.9 µs. Time your launches before you profile. After a profiler session has run in a process, every later launch in it costs 1.4 to 2.9 µs more.

A kernel that keeps the GPU busy for less time than its launch takes is paced by the CPU. The GPU finishes it and then waits for the next one.

#### Streams

A stream fixes the order of the kernels put into it. It does not reserve any SMs. Kernels in different streams are allowed to run at the same time, and they share the same 148 SMs. PyTorch puts everything on one stream per device unless code asks for another. To order work across streams, code records an event on one stream and waits for it from another.

Two kernels on two streams can run at the same time, block by block, as places on the SMs become free. A second stream does not make a pair of kernels faster when each of them alone already fills the SMs.

Measured on the B300 from a cool start, two 8192³ bf16 GEMMs took 1.236 ms on one stream and 1.235 ms on two. A GEMM and a copy of a 2 GiB tensor took 1.238 ms on one stream and 1.261 ms on two, where a full overlap would have taken 0.653 ms. Run back to back under the power cap, both pairs again came out within 2% of the one-stream time.

These runs used Halo's image, which sets `CUDA_DEVICE_MAX_CONNECTIONS=1`. That puts every stream of a process on one hardware queue, and NVIDIA's guide to the variable says this alone "can lead to GPU work serialization". So the totals show what two streams give under that setting, and they do not say whether the two kernels ran one after the other or shared the SMs. With the default of 8 queues the two pairs were not timed.

#### CUDA graphs

A **CUDA graph** records a sequence of kernels once and replays it with a single launch. Replaying a recording of 10,000 `add_` calls costs the CPU about 20 µs in all. Launching them one by one costs 48 ms. Both were measured on the B300, where the first replay of a recording took 9 ms. The GPU times in the table of the launch floor were measured this way, with the CPU out of the path.

A recording holds the shapes and the memory addresses it was made with, and it cannot contain control flow that depends on data. When shapes change from step to step, the recording does not fit the next step. Wrapping work in a graph can also add work, and [What fusion does not help with](#what-fusion-does-not-help-with) has a measured case.

> In Halo: `torch_compile_mode: reduce-overhead`, which captures CUDA graphs, ran 7% slower alone and 10% slower on top of the fused kernels of the Liger library on a mixture-of-experts model, where kernel shapes change every step. That is Qwen3-30B-A3B on 2 B300s with its experts split over both, 16,384-token rows, batch 1 ([torch.compile](../optimization/torch-compile.md)).

### The roofline

You now have two rates for this GPU, 1,388 TFLOP/s of arithmetic and 6.65 TB/s of memory traffic, and their ratio of about 210 FLOPs per byte. The roofline is a plot that tells you which of the two rates limits a given kernel.

The **arithmetic intensity** of a kernel, AI for short, is the FLOPs it does divided by the bytes it moves to and from HBM:

```text
AI = FLOPs / bytes moved to and from HBM        [FLOP/byte]
```

The bytes are everything the kernel reads plus everything it writes.

Now plot the rate a kernel can reach against its AI. At a low AI the kernel cannot run faster than memory delivers its bytes, so its rate is at most AI × bandwidth, a rising line. At a high AI its rate is at most the peak arithmetic rate, a flat line. The two lines make a roof, and the corner is called the **ridge**:

```text
attainable FLOP/s ≤ min(peak, AI × bandwidth)
ridge = peak / bandwidth = 1,388 × 10¹² / 6.65 × 10¹² ≈ 209 FLOP/byte      (B300, bf16, sustained)
```

The model comes from Williams, Waterman and Patterson (2009), and NVIDIA's guides call the same ratio the ops:byte ratio. With the burst GEMM peak in place of the sustained one, the ridge is about 270.

- A kernel whose AI is below the ridge is **memory-bound**. To speed it up, move fewer bytes per FLOP. More tensor cores will not help.
- A kernel whose AI is above the ridge is **compute-bound**. It needs faster arithmetic or a lower precision, and faster memory will not help.

Do not confuse memory-bound with out of memory. Memory-bound is about bandwidth, the rate at which bytes move. Running out of memory is about capacity, a separate limit with its own section.

The roof is an upper bound when its two rates are the true maxima. Drawn with the copy rate and the sustained 8192³ rate, as it is here, it is a reference line. Kernels that only read or only write run up to 15% above the memory line, and the 16384³ GEMM runs 6% above the flat one. A kernel can also sit far under the roof for other reasons, and this section meets the first of them.

#### The arithmetic intensity of a GEMM

For a bf16 GEMM, count each operand once and the output once, at 2 bytes per number:

```text
bytes = 2 × (M×K + K×N + M×N)
AI    = 2×M×K×N / bytes = M×K×N / (M×K + K×N + M×N)
```

No kernel could move fewer bytes for this GEMM, so this AI is an upper bound too.

A real kernel moves more. It computes the output in pieces, and for each piece it reads the rows of `X` and the columns of `W` that the piece needs. For the 8192³ GEMM in pieces of 128 rows by 256 columns that is 2,048 pieces at 6.29 MB each, 12.9 GB of reads against 268 MB of operands (computed). HBM cannot deliver 12.9 GB in the 0.61 ms the call takes. Most of those reads come from the L2 cache, where a piece finds the data its neighbors have just used. [Tiles and kernel selection](#tiles-and-kernel-selection) describes the pieces.

![X, M rows by K columns, times W, K by N, equals Y, M by N, drawn to one scale. X and Y grow with the tokens, W is fixed by the model. Under each block its bf16 bytes and under all three 2 × M × K × N FLOPs. Below, the product at K = N = 2880 at two sizes: at M = 128 W is 92% of the bytes, at M = 8,192 it is 15%.](../assets/diagrams/gemm_anatomy.png)

`M` is the dimension you control: batch size, row length and packing all change it. `K` and `N` are fixed by the model. The weight is read whatever `M` is, and only `X` and `Y` grow with `M`.

For a square weight, `K = N`, the formula becomes short:

```text
AI = M×K / (K + 2×M)
```

It tells you two things about a shape. However large `M` gets, the AI stays under `K/2`. And the AI reaches a ridge `r` at the row count

```text
M* = r×K / (K − 2×r)
```

Call `M*` the **crossover M** of the shape. With the unrounded ridge of 208.7 it is 244 rows for `K = N = 2880` and 232 rows for `K = N = 4096` (computed).

Work one row by hand, for `M = 128` and `K = N = 2880`:

```text
FLOPs = 2 × 128 × 2880 × 2880                = 2,123,366,400
bytes = 2 × (368,640 + 8,294,400 + 368,640)  = 18,063,360
AI    = 2,123,366,400 / 18,063,360           = 117.6 FLOP/byte
```

The weight alone is 16.6 MB, 92% of the bytes.

#### Why a small M is slow

The first measurement of this lecture comes from a sweep of `M` on the bf16 GEMM `[M, 2880] @ [2880, 2880]` (measured on one B300). The weight is read from HBM on every call. Each row was timed two ways. **Eager** means plain PyTorch calls, one after another. Device time is for the same calls replayed from a CUDA graph, so the CPU's launch cost is not in it. The roof column is min(AI × 6.65 TB/s, 1,388), computed.

| M | AI | Roof, TFLOP/s | Eager TFLOP/s (µs per call) | Device TFLOP/s (µs per call) | SM clock, MHz |
|---|---|---|---|---|---|
| 16 | 15.8 | 105 | 27.8 (9.56) | 32.4 (8.19) | 2,032 |
| 128 | 117.6 | 782 | 236.7 (8.97) | 353.8 (6.00) | 2,032 |
| 256 | 217.4 | 1,388 | 476.9 (8.90) | 623.6 (6.81) | 1,972 |
| 512 | 377.7 | 1,388 | 805.3 (10.55) | 808.3 (10.51) | 1,695 |
| 1,024 | 598.4 | 1,388 | 959.1 (17.71) | 997.2 (17.04) | 1,597 |
| 2,048 | 845.5 | 1,388 | 1,058 (32.1) | 1,065 (31.9) | 1,545 |
| 8,192 | 1,224.7 | 1,388 | 1,231 (110.4) | 1,227 (110.8) | 1,207, power cap |
| 32,768 | 1,379.4 | 1,388 | 1,339 (405.9) | 1,332 (408.1) | 1,207, power cap |

Read the device column first. From `M` of 24 to about 250 the time on the device does not depend on `M`: in a finer sweep it is 5.7 to 6.3 µs for every `M` from 24 to 232. The kernel takes as long for 24 rows as for 230, so the rate grows in proportion to `M`. The two smallest values are slower, 8.2 to 8.3 µs at 8 and at 16 rows on the same kernel as 24 rows.

That flat time grows with the weight. For four shapes the formula

```text
t ≈ 3.9 µs + weight bytes / 7.7 TB/s
```

fits it within about 0.5 µs:

| Weight `K × N` | Weight size | Measured device time at small M | Formula |
|---|---|---|---|
| 2048 × 768 | 3.1 MB | 3.93 µs | 4.31 µs |
| 2880 × 2880 | 16.6 MB | 6.06 µs | 6.05 µs |
| 4096 × 4096 | 33.6 MB | 8.76 µs | 8.26 µs |
| 8192 × 8192 | 134.2 MB | 21.15 µs | 21.33 µs |

For the 16.6 MB weight the constant term is two thirds of the call. For the 134 MB weight it is a fifth, and that GEMM sits near the rising line of the roofline. The roofline has no term for a cost that does not shrink with the work. This cost is the first thing in the lecture that keeps a kernel under the roof.

Treat the formula as a description of these four shapes and not as a mechanism. Three of the four are square, so they cannot tell a cost per byte from a cost that grows with `K`.

The time of a small weight also follows `K`. It is 2.5 µs at `K = 512`, in the table of [Rows per expert in an MoE layer](#rows-per-expert-in-an-moe-layer), and 4.8 µs at `K = 2880` with a narrow `N`, in the table of [Where the size of the grid does show](#where-the-size-of-the-grid-does-show). Two weights of the same 3.1 MB, `[2048, 768]` and its transpose, take about 4 and 2.9 µs. What sets the part that does not scale with the bytes was not measured. One candidate is the chain of steps along `K` that every block of the kernel runs one after another.

The 7.7 TB/s is a fitted slope. It is close to the rate of traffic that only reads, 7.2 to 7.3 TB/s, and above the 6.65 TB/s of a copy. An elementwise kernel with three reads and one write, `torch.addcmul`, measured 7.24 TB/s on the same GPU.

The weight has to come from HBM for these numbers to hold. If you call the GEMM in a loop on one weight, its 16.6 MB stay in the 126.5 MiB L2 cache. The calls then run at 1.2 to 1.34 times the rate at small `M`, 4.70 µs against 6.00 µs at `M = 128`.

A model reads a different weight in every layer, so the slower number is the one that applies to training. The benchmark rotates through many copies of its operands for that reason. If you benchmark one layer in a loop, you measure the faster one.

Now the eager column. Called from PyTorch, this GEMM cannot go faster than one call per 8.9 µs, the CPU's cost of launching it. Every eager point below `M` of about 512 therefore measures the CPU. For a smaller weight, `[M, 2048] @ [2048, 768]`, the same holds up to `M` of about 2,900.

Above the crossover M the kernel does not jump to the roof. It reaches 58% of the roof at `M = 512` and 88% at `M = 8,192`. Two things hold it down.

Near the ridge, the time that does not depend on `M` is still most of the call. At `M = 512` the call takes 10.5 µs, and 6.1 µs of that is the flat time measured at smaller `M`. The roofline takes the larger of two times, the one for the bytes and the one for the arithmetic, and a real kernel pays part of both.

At large `M` what is left is a cost of this particular `K = N = 2880`. From a cool start, at an equal clock of 2,032 MHz, this shape runs 1,559 TFLOP/s at `M = 8,192`, 13% below the 8192³ GEMM ([the width table](#the-measured-curve-has-teeth) has the comparison). Its best rate on the device in the whole sweep is 1,346 TFLOP/s.

The last column of the table explains a different gap, the one to the burst GEMM peak. The SM clock falls as the kernel does more arithmetic and the board nears its power limit.

![Log-log plot of rate against arithmetic intensity on one B300. A memory line, AI × 6.65 TB/s, meets a roof at 1,388 TFLOP/s at about 210 FLOP/byte, with a thin second roof at 1,793 for the first 40 ms. Three lines run under them. Two are a bf16 GEMM at K = N = 2880, timed on the device and timed from PyTorch, and the PyTorch line is lower at small M. The third, K = N = 8192, runs closest to the memory line, at 96% of it at M = 16. Dots mark the 8192³ GEMM on both roofs and the elementwise kernel addcmul far left. Four dots on the device line carry their share of the ceiling above them: 31% at M = 16, 45% at 128, 58% at 512, 88% at 8,192.](../assets/diagrams/roofline.png)

The rate between the rows of the table is not smooth. [The measured curve has teeth](#the-measured-curve-has-teeth) shows the whole curve.

#### Padding and packing

A pad token is a row of the GEMM like any other. Every linear layer computes it at full cost, and the result is discarded. Some attention kernels skip pad tokens. The linear layers never do.

**Packing** concatenates several documents into each row of the batch, so the same GEMMs carry tokens that count.

If three quarters of every row is padding, a packed batch holds 4 times the real tokens in the same shapes. The pad FLOPs therefore explain a gain of up to 4.

On one B300, 3.5 of that 4 arrived. Qwen3-4B trained on the same 16 documents of 1,024 tokens in two layouts: padded, as 16 rows of 4,096 tokens, and packed, as 4 rows' worth of 4,096. The benchmark fed the packed tokens flattened into one row of 16,384, with the positions restarting at each document, so the GEMMs see the same 16,384 positions that 4 rows would give them.

Both layouts ran with gradient checkpointing on. That setting saves memory by keeping only the input of each layer in the forward pass and computing the rest again in the backward pass, and [Gradient checkpointing](#gradient-checkpointing) gives its cost.

| Layout | Positions through the GEMMs | Step time | Real tokens/s | Peak memory |
|---|---|---|---|---|
| padded, 16 rows | 65,536 | 2,954 ms | 5,547 | 164.8 GiB |
| packed, 4 rows' worth | 16,384 | 841 ms | 19,484 | 58.1 GiB |

Packing ran 3.51 times the real tokens/s in 2.8 times less memory. It is 3.5 and not 4 because parts of the step cost the same in both layouts. Attention is one, since the run used a Flash Attention kernel, one of the kernels of [Attention](#attention), which skips pad tokens. The optimizer update and the launch of every kernel are the others. The run did not split the gap between them. With checkpointing off, the padded batch did not fit in memory, and the packed one ran at 25,144 tokens/s.

![Bars for one training step of Qwen3-4B on a B300 on the same 16 documents of 1,024 tokens, in three columns: positions in the step, step time and real tokens per second. Padded to 16 rows of 4,096 with gradient checkpointing: 65,536 positions, 49,152 of them padding, 2,954 ms, 5,547 tokens per second. Packed into 4 rows: 16,384 positions, 841 ms, 19,484. Without checkpointing the padded layout runs out of memory and the packed one takes 652 ms.](../assets/diagrams/padding_against_packing.png)

A recorded run on a larger model gained more. On Qwen3-30B-A3B, packing ran 13,341 real tokens/s per GPU against 922 for unpacked rows, 14.5 times. The run was on 2 B300s with the experts split over both, an older Flash Attention kernel (FA2), documents of about 1,024 tokens, batch 2 per GPU and gradient checkpointing on ([Padding-Free Collator](../optimization/padding-free-collator.md#benchmark-results)).

Little of that gain is padding. The unpacked rows of that run were padded only to the longer document of each pair, and the packed rows were 4,096 tokens long, about four documents each. The packed step therefore holds 4.1 times the real tokens, and it is also 3.5 times shorter (computed from the rates and step times in its table: 922 × 2.08 s against 13,341 × 0.59 s). The page that holds the run gives no cause for the shorter step.

So pack before anything else: pad tokens are GEMM rows you pay for and throw away. Then check how far the `M` of the step's largest GEMMs is above the crossover M, and if it is not far, use longer packed rows or a bigger batch.

> In Halo: packing and its alternatives are set through the collator ([Padding-Free Collator](../optimization/padding-free-collator.md)).

Check yourself:

1. You time an 8192³ `a @ b` with no `synchronize` and read about 30 µs. What did you measure?
2. What is the AI of `[64, 4096] @ [4096, 4096]` in bf16?
3. For a square weight, `AI = M×K / (K + 2×M)`. At what `M` does `K = 2048` reach a ridge of 209, and what is the highest AI this shape can reach?
4. Use the formula of this section, 3.9 µs plus the weight's bytes at 7.7 TB/s, to estimate the device time for a 6144 × 6144 bf16 weight and a few dozen rows. How far do you trust the estimate?

[Answers](#answers-on-the-roofline) are at the end of the page.

### Tiles and kernel selection

A tensor-core GEMM kernel cuts the output `Y` into rectangles called **tiles**. In the plain design, one block computes one tile.

NVIDIA's matrix-multiplication guide describes what follows when the tile has a fixed size. It is written for an earlier version of **cuBLAS**, NVIDIA's matrix library and the code that runs when PyTorch executes `a @ b`, on an A100 GPU with 108 SMs. That version had eight tile sizes, from 256 × 128 down to 64 × 64.

**Tile quantization** is the cost of a dimension that is not a multiple of the tile, which leaves the last tile partly empty. At `M = 130` with 128-row tiles the second tile holds 2 rows of 128. **Wave quantization** is the cost of a tile count just past a multiple of the SM count. With one block per SM, 149 tiles on 148 SMs need a second wave for one tile. On the guide's own curve the rate about halves between `N = 1536` and `N = 1544`.

The cuBLAS on a B300 behaves differently. It holds many GEMM kernels. Their names carry the tile size, so a profiler trace shows which one ran. A name gives the columns first: a kernel named `…tst_160x64…` uses tiles of 64 rows by 160 columns. This page prints rows × columns. Across a sweep of 576 values of `M` at `K = N = 2880` cuBLAS uses 55 different kernels and changes from one to another 145 times (measured on one B300):

| M | Most common tile, rows × columns of `Y` |
|---|---|
| 128 | 64 × 64 |
| 200 to 424 | 80 × 64 up to 144 × 64 |
| 408 to 1,000 | 80 × 128 up to 160 × 128 |
| from about 2,880 | 256 × 128 or 128 × 256 |

cuBLAS shrinks the tile at small `M`, and that keeps the grid large. For every `M` from 40 to 432 the launch has 92, 120 or 138 blocks, on a GPU with 148 SMs. At `M = 128` the tile is 64 × 64, which cuts this output into 2 × 45 = 90 tiles. The grid has 92 blocks, two more than the tiles.

So a small `M` does not mean a small grid. The number of blocks does not set the time either: 92 blocks at `M = 64` take 5.93 µs and 138 blocks at `M = 96` take 5.83 µs. The time follows the shape of the weight, as in the previous section.

#### Where the size of the grid does show

The size of the grid shows in one sweep that was built to find its effect. Hold `M = 128` and `K = 2880` and widen `N` (measured on one B300):

| N | Device time | Blocks in the grid | Tile, rows × columns |
|---|---|---|---|
| 128 | 4.51 µs | 24 | 32 × 64, each tile split in three along `K` |
| 512 | 4.84 µs | 32 | 32 × 64 |
| 1,024 | 4.99 µs | 64 | 32 × 64 |
| 2,048 | 5.38 µs | 128 | 32 × 64 |
| 4,736 | 7.32 µs | 148 | 64 × 64 |
| 9,344 | 12.40 µs | 146 | 64 × 128 |
| 18,560 | 21.86 µs | 146 | 64 × 256 |

While the tile count is under 148, more columns mean more blocks and little more time. Between `N = 128` and `N = 2048` the call does 16 times the work in 1.2 times the time. The likely reason is that the added blocks run on SMs that had none. At `N` = 128, 256 and 384, the three smallest values of the sweep, the call runs two kernels: cuBLAS splits each tile along `K`, and a second kernel adds the parts. From `N = 512` on it runs one.

The grid climbs twice, as the figure shows. At `N = 2432` cuBLAS doubles the rows of the tile and the grid falls from 144 blocks to 76, for a step in time from 5.49 to 5.87 µs. From `N = 4736` on it widens the tile each time the grid would pass 148 blocks, and the time grows with the weight. Over the whole stretch up to `N = 4736`, 37 times the work takes 1.6 times as long.

![Two panels against N from 128 to 18,944 for a bf16 GEMM at M = 128, K = 2880 on one B300. Above, the time per call on the device: 4.5 µs at N = 128, 7.3 µs at 4,736, then up in steps to about 22 µs. Below, the blocks in the grid: up to 144, down to 76 at N = 2,432, up to 148, the number of SMs, at 4,736, then between 104 and 148. The three smallest N, two-kernel calls, are hollow.](../assets/diagrams/tile_fill.png)

#### The measured curve has teeth

The curve of rate against `M` is a sawtooth, and the cause is the library's choice of kernel. Stepping `M` by 8 at `K = N = 2880`, every drop of 8% or more between neighbors falls where cuBLAS switches kernel. The largest is 14.3%, from 400 to 408 rows.

The rate of a GEMM depends on which kernel cuBLAS picks for its shape. That choice moves the rate by up to 14% from one shape to its neighbor, and it can leave a shape 11 to 17% below the best rate at a smaller `M`, as the table shows. So measure your own shape.

| Range of `M` | Worst shortfall against the best rate at any smaller `M` |
|---|---|
| up to 512 | 17% |
| 512 to 2,048 | 11% |
| 2,048 to 16,384 | 14.5% |

![Two panels against M from 8 to 40,960 on a log axis for a bf16 GEMM at K = N = 2880 on one B300, 576 points. The upper panel is the rate on the device, with a tick at each of the 145 kernel changes, a line down to each of the 9 drops of 8% or more, a dashed line at the crossover M of 244 and, above it, the tile at seven values of M. The lower panel is the SM clock: first below 2,032 MHz at M = 176, below from M = 440 on, falling towards 1,200.](../assets/diagrams/gemm_kernel_selection.png)

Those points ran back to back, so from a few hundred rows up the falling SM clock is mixed into them. To separate the two effects, a second pass over `M` from 1,024 to 2,048 started every point from a cool GPU at 2,032 MHz.

The teeth are still there. The rate moves between 1,209 and 1,502 TFLOP/s with a median of 1,386. The deepest dip, at `M = 1208`, is 14% below its neighbors, and cuBLAS picks a tile of 128 rows by 256 columns there. Run back to back, the same points are 1.23 to 1.35 times slower, and that factor is the power cap alone.

Wave quantization does not show here as a rule. It would make launches with more blocks than SMs slower. In the cool-start pass, launches of 120 to 144 blocks have a median of 1,395 TFLOP/s, launches of 240 to 264 blocks 1,347 and launches of 288 blocks 1,434, with a different `M` in each group.

Tile quantization does not show either. It would make a call slower just past a tile boundary. Stepping `M` by 1 across 128, 256, 1,024 and 4,096, the time moves by under 1% from one row count to the next on either side of the boundary. Just past each power of two cuBLAS switches kernel and the call gets slightly faster: 6.17 to 5.96 µs from 128 to 129 rows and 17.23 to 16.43 µs from 1,024 to 1,025.

The width matters by about a tenth. Some cuBLAS kernels on this GPU run their blocks in pairs called clusters, and [Inside a tensor-core GEMM kernel](#what-each-gpu-generation-changed) explains what a pair shares. These four GEMMs ran at `M = 8,192` from a cool start, all at 2,032 MHz (measured on one B300):

| `K = N` | Width in column tiles of 128 | Blocks in the grid | Rate | Kernel cuBLAS picked |
|---|---|---|---|---|
| 2816 | 22 | 704 | 1,720 TFLOP/s | a two-block cluster kernel |
| 2880 | 22.5 | 736 | 1,559 TFLOP/s | a kernel without clusters |
| 2944 | 23 | 736 | 1,595 TFLOP/s | a kernel without clusters |
| 3072 | 24 | 768 | 1,648 TFLOP/s | a two-block cluster kernel |

The 8192³ GEMM measured the same way runs at 1,793 TFLOP/s, so a width of 2,880, which the gpt-oss models use, is 13% below a large square. A partly filled tile does not explain the table. A width of 2,944 is a whole 23 tiles and still runs the slower kind of kernel. In these four points the cluster kernel ran where the number of column tiles is even.

#### Multiples of 8

A bf16 GEMM `[M, K] @ [K, N]` slows down several times over when `K` or `N` is not a multiple of 8. These slowdowns were measured on one B300 at `M = 4096`, stepping one dimension at a time through two windows, one around 2,880 and one around 4,096:

| Dimension off a multiple of 8 | Window at 2,880 | Window at 4,096 |
|---|---|---|
| `K`, even value | 4.7 to 4.8 times slower | 4.4 to 4.6 times slower |
| `K`, odd value | 7.1 to 7.4 times | 6.8 to 7.2 times |
| `N`, even value | 4.4 times | 4.4 to 5.0 times |
| `N`, odd value | 7.9 to 8.3 times | 7.1 to 8.7 times |
| `M`, any value | no slowdown (0.99) | no slowdown (0.99 to 1.04) |

At `K = 4096` the GEMM takes 107 µs, and at `K = 4095` it takes 724 µs. The aligned points ran under the power cap at 1,280 to 1,500 MHz and the slow points at the full 2,032 MHz, so at equal clocks the gap is larger than these ratios.

NVIDIA's matrix-multiplication guide asks for matrix dimensions "aligned to multiples of 16 bytes", which is 8 bf16 numbers or 4 fp32 numbers. The guide names `M`, `N` and `K` alike. On this GPU the 16 bytes hold for `K` and `N`, and `M` is free in every form of the GEMM that was timed.

The fp32 measurements agree. With TF32, a width off a multiple of 4 runs 4.0 to 4.3 times slower. Full-precision fp32 on the CUDA cores has no such cliff. It only runs 1.07 to 1.10 times slower once `M` or `N` passes 4,096.

Beyond 8 there is little left to gain. With `K` at 2,888 and 2,896 the GEMM runs within 1% of 2,880. In the window at 4,096, widths that are multiples of 8 and not of 16 run 3 to 9% slower than multiples of 64.

![Three panels, bf16, fp32 with TF32 and full-precision fp32, plotting the time of a GEMM at M = 4,096, K = N = 2880 against its aligned time as K, N or M grows by 0 to 16. In bf16, K or N off a multiple of 8 runs 4.4 to 4.8 times slower when even and 7.1 to 8.3 times when odd. With TF32 the step is 4 and the slowdown 4.0 to 4.3 times. M has no effect in bf16 or with TF32. Full-precision fp32 stays at 1 for K and N, and its M line sits at 1.10.](../assets/diagrams/alignment_slowdown.png)

Your token count is the one dimension you change from step to step, so the backward pass deserves the same test. wgrad, `Xᵀ @ dY`, sums over the `M` rows, which makes the row count its inner dimension.

Timed the same way around 4,096, a row count off a multiple of 8 costs it nothing: 1.00 to 1.01 times for every `M` from 4,096 to 4,128. Its slowdown stays on `K` and `N`: 4.2 to 4.4 times for an even value off a multiple of 8 and 7.0 to 7.7 times for an odd one (measured on one B300).

So in the three forms timed, the two dimensions of the weight have to be multiples of 8 and the row count is free. The grouped GEMM of [Many small kernels](#many-small-kernels) is the exception on this page: there a row count off a multiple of 8 costs up to 1.23 times, and routing sets those counts.

> In Halo: kernels whose `K` or `N` follows the data round it to a multiple of 8 ([Flash Attention → Masked matmul attention](../optimization/flash-attention.md#masked-matmul-attention)). Row lengths are left unpadded. On one run padding them to a multiple of 8 did not pay: it read 4% slower for 0.043% more tokens (recorded on Halo 1.2.0, a full fine-tune of gpt-oss-20b with its experts over 2 GPUs, 3 replicas and unpacked rows of about 7,950 tokens).

#### Rows per expert in an MoE layer

A **mixture-of-experts** layer, MoE for short, replaces one MLP with `E` smaller MLPs called experts. A router sends each token to `k` of them. This lowercase `k` is a count of experts and has nothing to do with the width `K`. Many current models are built this way, and their GEMMs are small even when your batch is large. This is the small `M` of [Why a small M is slow](#why-a-small-m-is-slow) again, with a cause that sits in the model.

On one GPU each expert's GEMM has about

```text
M ≈ T × k / E
```

rows. With 16,384 tokens, 128 experts and 8 experts per token that is 1,024 rows. With 256 experts it is 512.

| Model | Experts `E` | Experts per token `k` | Rows per expert as a share of `T` | Width of an expert's GEMMs |
|---|---|---|---|---|
| gpt-oss-20b | 32 | 4 | 1/8 | 2,880 |
| gpt-oss-120b | 128 | 4 | 1/32 | 2,880 |
| Qwen3-30B-A3B | 128 | 8 | 1/16 | 2,048 and 768 |
| Qwen3.6-35B-A3B | 256 | 8 | 1/32 | 2,048 and 512 |

gpt-oss-120b gives each expert the same share of the tokens as Qwen3.6. The families differ in width, and a narrow expert is a small weight. These three narrow shapes were timed on the device (measured on one B300, weight read from HBM):

| M | `K = 2048, N = 1024` | `K = 512, N = 2048` | `K = 2048, N = 768` |
|---|---|---|---|
| 16 | 3.95 µs, 17 TFLOP/s | 2.51 µs, 13 TFLOP/s | 4.09 µs, 12 TFLOP/s |
| 128 | 4.17 µs, 129 TFLOP/s | 2.72 µs, 99 TFLOP/s | 4.10 µs, 98 TFLOP/s |
| 512 | 4.59 µs, 468 TFLOP/s | 3.47 µs, 310 TFLOP/s | 4.34 µs, 371 TFLOP/s |
| 2,048 | 8.73 µs, 984 TFLOP/s | 6.12 µs, 702 TFLOP/s | 7.57 µs, 851 TFLOP/s |
| 8,192 | 29.67 µs, 1,158 TFLOP/s | 18.52 µs, 928 TFLOP/s | 24.72 µs, 1,043 TFLOP/s |

At small `M` these kernels take 2.5 to 4.1 µs, and the time follows `K`: 2.5 µs at `K = 512` and about 4 µs at `K = 2048`. At `M = 8,192` they reach 63 to 79% of the best sustained rate measured on any shape, 1,472 TFLOP/s. Called from PyTorch, every one of these GEMMs costs the CPU about 9 to 12 µs up to `M` of 2,048, which is more than the kernel takes. A Python loop over experts is therefore paced by the CPU across this whole range.

Check yourself:

1. Your model has a hidden size of 2,050. What do you expect of its bf16 GEMMs, and why does the same model in full-precision fp32 not care?
2. Is 2,880 a multiple of 8? Of 128?
3. Two bf16 GEMMs differ by 8 rows and nothing else, and one runs 14% slower. What is the likely cause, and how do you confirm it?
4. A step has 8,192 tokens on one GPU. How many rows does each expert's GEMM have on gpt-oss-120b, with 128 experts and 4 per token? Where does that put it against the crossover M of 244 for its width of 2,880?

[Answers](#answers-on-tiles-and-kernel-selection) are at the end of the page.

### Inside a tensor-core GEMM kernel

Every fast kernel in the rest of this lecture is built the way a GEMM kernel is, so it helps to open one. The kernel literature uses the vocabulary you meet inside: MMA, TMA, stage and epilogue.

One block computes one output tile, and it gets there in steps along `K`. In each step the block loads a strip of `X` and a strip of `W`, multiplies them on a tensor core and adds the product to a running result called the accumulator. After the last step it writes the tile out.

The load is done by the **Tensor Memory Accelerator**, TMA. It is a hardware unit that copies a rectangular piece of a tensor between HBM and shared memory on its own, once a single thread has started it. GPUs have had it since the Hopper generation. Its name is close to "tensor memory", and the two are different things: TMA copies data, and tensor memory stores it.

The multiply is one tensor-core instruction, called an **MMA** for matrix multiply-accumulate.

Between the two sits a pipeline. The kernel keeps several shared-memory buffers, called stages, so that the load for a later step runs while the tensor cores multiply the current one. More stages need more shared memory. Some warps of the block only start loads and the other warps only compute, an arrangement called warp specialization. The pipeline is how a GEMM kernel keeps working while memory is slow, in place of the many resident warps an ordinary kernel relies on.

The last part of the kernel is the **epilogue**. It reads the accumulator out and writes the tile to HBM. Libraries let more work be attached at this point, which is called a fused epilogue. cuBLAS offers a set of them through its cuBLASLt interface, and **CUTLASS**, NVIDIA's open-source kernel templates, lets the author write the epilogue.

![Three parts. A band of X and a band of W, each cut into three strips along K, make one tile of Y. A grid of lanes over five time slots shows loads filling two shared-memory stages while the multiply-accumulate reads the other, the accumulator filling, and the epilogue writing the tile. Three cards name what Hopper and Blackwell share and where they differ.](../assets/diagrams/gemm_kernel_pipeline.png)

The figure is a schematic with three steps and two stages. Real kernels use more of both.

#### What each GPU generation changed

The two current generations differ in the multiply step and in where the accumulator lives. The Hopper column is NVIDIA's specification:

| | Hopper (H100, H200) | Blackwell (B200, B300) |
|---|---|---|
| Tensor-core instruction | `wgmma`, fourth generation | `tcgen05.mma`, fifth generation, on one SM or on two |
| Started by | a group of 4 warps | a single thread |
| Unit for bulk loads | TMA | TMA |
| Accumulator | registers | tensor memory, 256 KiB per SM |
| Low-precision inputs | fp8 | fp8, fp6 and fp4, plain or block-scaled ([Number formats](#number-formats) explains the block-scaled names mxfp8, mxfp4 and nvfp4) |
| HBM bandwidth | 3.35 TB/s on an H100, 4.8 TB/s on an H200 | 8 TB/s by specification, 6.6 TB/s measured on a B300 for a copy |

On Blackwell the largest MMA step covers a tile of 128 × 256 outputs and, for bf16 inputs, 16 positions along `K`. The fp32 accumulator for that tile is 128 × 256 × 4 bytes = 128 KiB, half of an SM's tensor memory. The instruction has two forms: one runs on a single SM, and in the other two blocks on two SMs run one MMA together. The epilogue reads tensor memory out with a separate instruction, `tcgen05.ld`.

The two-SM form needs a **cluster**, a group of blocks that the GPU guarantees to run at the same time inside one group of SMs, which NVIDIA calls a GPC, a GPU processing cluster. Clusters exist since Hopper, and Blackwell adds an MMA that spans the two blocks of one.

The names of cuBLAS's kernels show two-block cluster kernels at many shapes, small ones among them: the kernel it picks for 128 rows at `K = N = 2880` is one. In the width table of the previous section, the widths 2,816 and 3,072 ran a cluster kernel and 2,880 and 2,944 ran one without.

#### The accumulator is fp32

A bf16 GEMM on the tensor cores adds its products up in fp32 and rounds once, when it writes the result. A measurement on the B300 agrees for cuBLAS. A bf16 dot product was computed as a `[1, K] @ [K, 1]` matmul with 1,024 to 16.7 million terms. Read out in fp32, it stays within a relative error of 6 × 10⁻⁸ of the exact sum. Read out in bf16, it is off by up to 3.4 × 10⁻³, which is the one final rounding.

The source of PyTorch's grouped GEMM, a kernel that the next section introduces, declares the same thing: bf16 operands and an fp32 accumulator.

Do not read this as a property of every tensor-core kernel. DeepSeek-V3 reports that fp8 GEMMs on its H800 GPUs accumulate with about 14 bits.

#### A kernel is written for one generation

The instruction, the place where the accumulator lives, the tile sizes and the scheduling all change from one generation to the next. An older kernel still runs on a newer GPU, and it leaves most of the newer tensor core unused.

FA2 and FA4 are two attention kernels that compute the same result. FA2 was written for GPUs two generations before Blackwell and FA4 for Blackwell. On a B300, FA4 is 3.6 to 3.9 times faster: 0.70 ms against 2.71 ms for the forward pass, and 3.0 ms against 10.9 ms for forward plus backward. Those times are recorded ones for the kernel alone, at batch 2, 8,192 tokens and 32 heads of width 128 ([Flash Attention](../optimization/flash-attention.md#fa4-vs-fa2-vs-sdpa-on-blackwell)).

> In Halo: FA4 is selected on Blackwell and FA3 on Hopper. The Blackwell image carries FA2 as well, and several model families run on it ([Flash Attention](../optimization/flash-attention.md)).

### Many small kernels

When each kernel does less work on the GPU than its launch costs on the CPU, the speed of the step is set by how fast the CPU launches kernels. Such a step is **overhead-bound**. The name comes from Horace He's essay in the reading list; other Halo pages write host-bound or launch-bound for the same thing.

The second measurement of the opening is the clearest case. An MoE layer of Qwen3-30B-A3B has 128 experts, and written as a Python loop it runs one matmul per expert for each of its projections. With 4,096 routed GEMM rows, 32 for each expert, each matmul is `[32, 2048] @ [2048, 768]`. On one B300 the 128 matmuls of one projection take 1.12 ms, which is 8.75 µs each (recorded in Halo's benchmark, median of 50 runs).

Those 8.75 µs are the CPU's cost of launching one `torch.mm`. The kernel itself keeps the GPU busy for about 4 µs at this shape, as the table of narrow experts shows. For more than half of every call the GPU waits for the CPU.

**Grouped GEMM** does the same work in one launch. It is one kernel that runs many matmuls with different row counts. In PyTorch it is `torch.nn.functional.grouped_mm`, built from CUTLASS, and it handles bf16 only. The same 128 matmuls take 0.11 ms as one grouped GEMM ([Grouped GEMM](../optimization/grouped-gemm.md)). An expert MLP of this model has three projections, so a real layer runs 384 matmuls as a loop and 3 launches when grouped. Some models store two of the three as one matrix and have 2 projections.

![Two timelines on one time scale, each with a CPU lane and a GPU lane. In the first, 128 launches of 8.75 µs fill 1.12 ms on the CPU lane, with 128 kernels of 4.10 µs and gaps between them on the GPU lane. An enlarged stretch shows three launches, each over a 4.10 µs kernel and 4.65 µs of idle GPU. In the second, one grouped launch is a single block of 0.11 ms.](../assets/diagrams/launches_against_work.png)

A rerun of that benchmark for this lecture shows two more things (one B300, one projection, forward pass):

| Expert shape | Rows per expert | Loop | Grouped GEMM | Ratio |
|---|---|---|---|---|
| 128 experts, `[M, 2048] @ [2048, 768]` | 32 | 1.25 ms | 0.11 ms | 11.6 |
| the same | 256 | 1.24 ms | 0.12 ms | 10.2 |
| 256 experts, `[M, 2048] @ [2048, 512]` | 128 | 2.52 ms | 0.14 ms | 17.8 |
| the same | 256 | 2.52 ms | 0.16 ms | 16.1 |

The loop takes the same time for 32 rows per expert as for 256, about 9.8 µs per matmul, because the CPU sets its pace. And its time doubles with the number of experts while the grouped launch barely moves. The 9.8 µs here and the 8.75 µs of the recorded run are months of software apart, and the launch floor also moves from one process to another.

Grouping helps memory as well, through a different mechanism. The loop saves one small tensor per matmul for the backward pass, and the grouped call saves one per projection. On Qwen3.6, with 256 experts and 2 projections, 512 saved tensors become 2. The bytes are about the same, and the saving was not measured. Many small buffers of changing sizes fragment the allocator's pools, and one buffer does not.

On a whole training step the gain depends on how many experts a GPU holds and how many rows each one gets. A grouped kernel uses one tile size for all of its matmuls, where a loop lets the library pick a kernel for each shape. The lead of grouping therefore narrows as the rows per expert grow.

In Halo's benchmarks grouped GEMM ran 31% to 78% faster than the loop on gpt-oss-20b, with its 32 wide experts, and 137% to 310% faster on two Qwen models with 128 and 256 narrow ones. Those runs had the experts split over 2 to 8 GPUs ([Grouped GEMM → Grouped vs the loop path](../optimization/grouped-gemm.md#grouped-vs-the-loop-path)).

Row counts off a multiple of 8, which routing produces at every step, cost a grouped GEMM at most 1.23 times, in a separate measurement on GLM-5.3-Flash's 4,096-wide `gate_up` product (recorded on Halo 1.2.0, [Grouped GEMM](../optimization/grouped-gemm.md#performance)). The note in [Multiples of 8](#multiples-of-8) reports what padding row lengths to a multiple of 8 did on one run.

You can recognize an overhead-bound step from outside, and power is the clearer sign. On the B300 a loop of 1-element kernels read 52% utilization and 241 W of the 1,100 W limit, and a loop over 128 experts with two small matmuls each read 59% and 374 W. Utilization alone is weaker: the 128-row GEMM loop of the opening is paced by the CPU too, and it reads 93%. In a profile you see thousands of kernels that each take a few microseconds.

A step like that has two remedies. One is fewer launches: put the work into one kernel, or replay a CUDA graph where the shapes are fixed. The other is less CPU work between launches.

> In Halo: grouped GEMM is the default for MoE expert compute ([Grouped GEMM](../optimization/grouped-gemm.md)).

### Memory-bound kernels and fusion

Most of the work in a model that is not a matmul does a few FLOPs for each value it touches. Normalization layers, activation functions such as SiLU and GELU, residual adds, position encodings, the softmax and the optimizer update are all like this. Their AI is of order 1 FLOP/byte, two orders of magnitude under the ridge. The time of such a kernel is the time it takes to move its bytes.

Measured on 8192 × 8192 bf16 tensors on one B300, an add, an RMSNorm, a SiLU-and-multiply and an AdamW update have an AI of 0.2 to 1.1 FLOP/byte, computed from a hand count of their FLOPs. All of them move 5 to 6.7 TB/s, and all of them do 1 to 6 TFLOP/s of arithmetic, 0.1 to 0.4% of the sustained GEMM peak.

The worked example is `silu(a) * b` on two bf16 tensors of 8192 × 8192 numbers, 134.2 MB each (measured on one B300):

| Variant | Kernels | Passes over a tensor | Bytes moved | Time | Effective rate |
|---|---|---|---|---|---|
| `F.silu(a) * b`, eager | 2 | 5 | 671 MB | 0.116 ms | 5.79 TB/s |
| `a * torch.sigmoid(a) * b`, eager | 3 | 8 | 1,074 MB | 0.177 ms | 6.07 TB/s |
| one hand-written fused kernel | 1 | 3 | 403 MB | 0.060 ms | 6.71 TB/s |
| `torch.compile`, default mode | 1 | 3 | 403 MB | 0.064 ms | 6.32 TB/s |

Count the passes yourself. `F.silu(a)` reads `a` and writes an intermediate tensor. The multiply reads the intermediate and `b` and writes the result. Together they make five passes of 134.2 MB. A single kernel can read `a` and `b`, keep the intermediate value in registers and write the result: three passes. Combining operations into one kernel so that intermediates never go to HBM is called **fusion**.

![y = silu(a) × b on two 8,192 × 8,192 bf16 tensors, drawn as passes between HBM and kernels. Eager, two kernels: the first reads a and writes an intermediate s, the second reads s and b and writes y. Five passes, 671 MB, 0.116 ms. Fused, one kernel reads a and b, keeps s in registers and writes y. Three passes, 403 MB, 0.060 ms.](../assets/diagrams/fusion_roundtrips.png)

The fused kernel is 1.93 times faster for 1.67 times fewer bytes. The rates in the last column are all near HBM bandwidth, which tells you these kernels do nothing but move memory.

Compare a GEMM of the same footprint. An 8192³ bf16 GEMM also reads two tensors and writes one, 403 MB, and it does 1.1 × 10¹² FLOPs on them. It takes 0.77 to 0.79 ms sustained and 0.61 ms on a cool GPU. The GEMM takes 5 to 7 times as long as the eager `silu(a) * b` and does about 3,000 times the arithmetic. The elementwise work is 6.7 × 10⁷ values at about 5 FLOPs each, 3.4 × 10⁸ FLOPs (computed).

#### A fused kernel in seven lines

Fused kernels of this kind are short. This one is written in **Triton**, a Python language for GPU kernels (the name has nothing to do with NVIDIA's Triton Inference Server). It is reduced from the fused activation Halo uses, and the lines after the kernel build two tensors and launch it:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def swiglu(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # this block's elements
    mask = offs < n                                        # the last block is partly empty
    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, gate * tl.sigmoid(gate) * up, mask=mask)

gate = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
up = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(gate)
n = gate.numel()

swiglu[(triton.cdiv(n, 1024),)](gate, up, out, n, BLOCK=1024)  # grid = number of blocks
```

On the B300 this kernel takes 0.059 ms on these tensors, the same as the hand-written kernel in the table and 1.97 times faster than the eager pair (measured). Its output is also closer to an fp32 reference than the eager one, a relative error of 1.66 × 10⁻³ against 2.41 × 10⁻³. It rounds to bf16 once, and the eager pair rounds the intermediate as well.

Everything from the section on how a kernel runs is in it. The last line launches a grid, one block per 1,024 elements. `BLOCK` is a number of elements, and how many threads work on them is Triton's choice. `program_id` tells a block which one it is. The mask covers the last block, which is only partly filled. The two loads bring bf16 values from HBM and widen them to fp32 in registers, and the one store writes the result back.

#### Predicting a memory-bound kernel

You can predict the time of a memory-bound kernel from its bytes alone. Take the AdamW update in bf16. It reads the weight, the gradient and two running averages, and it writes the weight and the two averages back: 4 reads and 3 writes of 2 bytes, 14 bytes per parameter. With 14 × 10⁹ trainable parameters that is 196 GB, and at 6.65 TB/s it takes 29.5 ms (computed).

Two measurements come out somewhat above the count. On Qwen3-8B, with all 8.19B parameters trained, the count gives 115 GB and 17.2 ms. Halo's fused AdamW step, AdamWBF16, is written as one pass. It took 20.6 ms on one B300, which is 5.6 TB/s and 20% above the count (measured). On gpt-oss-20b, with about 14B trainable parameters, it took 31.8 ms, 8% above the count (recorded, [BF16 Optimizer](../optimization/bf16-optimizer.md#benchmarks)).

PyTorch's own fused AdamW took 36.0 ms on Qwen3-8B, which is 3.2 TB/s, and 62.0 ms on gpt-oss-20b. It is another kernel on this page that is slower than the memory under it, like `fill_` and `torch.sum` in [Memory and its bandwidth](#memory-and-its-bandwidth). Why it is that slow was not investigated.

#### Where fused kernels come from

Fused kernels reach a training run from four places. Liger is a library of hand-written ones, with Triton kernels for RMSNorm, rotary position encoding, the SwiGLU activation and cross-entropy. PyTorch ships some itself, such as a fused `F.rms_norm`. `torch.compile` generates them by fusing chains of elementwise operations, and the fourth row of the `silu(a) * b` table is its output. For the places none of these cover, you write your own.

On a whole step these kernels take a quarter to a third off the time. Measured on Qwen3-4B on one B300, with a forward and a backward pass per step, Liger's kernels for the norm, the activation, the position encoding and cross-entropy changed the step as the table shows. The cross-entropy kernel of this run still builds the full table of output scores, the logits that [The fused loss](#the-fused-loss) describes, and the fused loss of that subsection is off.

| Row length | Gradient checkpointing | Step, plain PyTorch | Step, fused kernels | Peak memory, plain and fused |
|---|---|---|---|---|
| 4,096 | off | 178.5 ms | 133.3 ms | 39.09 and 27.28 GiB |
| 16,384 | off | 813.9 ms | 576.6 ms | 133.82 and 86.57 GiB |
| 4,096 | on | 231.2 ms | 151.6 ms | 17.62 and 17.62 GiB |
| 16,384 | on | 1,045.5 ms | 726.5 ms | 43.10 and 29.03 GiB |

The step is 25 to 34% shorter. Without gradient checkpointing, a third of the memory held for the backward pass goes too, because a fused kernel leaves no intermediates for autograd to keep. With checkpointing on at 4,096 tokens the peak does not move, and what sets it there was not measured.

Halo's benchmarks of whole training runs agree. With Liger's kernels and its cross-entropy kernel, which still builds the logits, Qwen3-8B on one B300 at 16,384-token rows ran 15,966 tokens/s against 11,746, a gain of 36%. Qwen3-30B-A3B, with its experts over 2 GPUs, 8,192-token rows and batch 4, gained 39% with the same kernels and 31% with the fused loss in place of the cross-entropy kernel. Both ran with gradient checkpointing on ([Liger Kernels](../optimization/liger-kernels.md#benchmarks)).

> In Halo: Liger's kernels are applied per model family, a fused activation runs inside the expert-parallel MoE layer, and the optimizer step is one fused pass. `torch.compile` is opt-in and off by default ([Liger Kernels](../optimization/liger-kernels.md), [torch.compile](../optimization/torch-compile.md)).

#### The fused loss

The last step of a language model multiplies the hidden state of every position by the output matrix. The result is called the logits, one score per vocabulary entry per position. For 32,768 positions and the 151,936 entries of Qwen3's vocabulary that is 4.98 × 10⁹ scores, 9.96 GB in bf16. The loss then makes an fp32 copy of 19.9 GB (computed).

Measured at the level of the kernels, on one B300 with a hidden size of 4,096, the plain loss allocates much more than the logits themselves:

| Positions | Logits in bf16 | Plain loss, forward | Plain loss, forward and backward | Fused loss, forward and backward |
|---|---|---|---|---|
| 4,096 | 1.16 GiB | 4.64 GiB, 6.1 ms | 6.96 GiB, 15.0 ms | 1.23 GiB, 29.9 ms |
| 16,384 | 4.64 GiB | 18.55 GiB, 24.6 ms | 27.82 GiB, 60.5 ms | 1.43 GiB, 56.7 ms |
| 32,768 | 9.27 GiB | 37.09 GiB, 50.2 ms | 55.64 GiB, 124.2 ms | 1.70 GiB, 96.0 ms |
| 65,536 | 18.55 GiB | 74.19 GiB, 103.9 ms | 111.28 GiB, 254.1 ms | 2.24 GiB, 180.2 ms |

The memory columns are the allocation above the inputs. The plain loss holds two fp32 tensors in the forward pass, the copy and its log-softmax, which is 4 times the bf16 logits. Through the backward pass it holds 6 times.

A **fused linear cross-entropy** loss avoids building the logits. It is a loop over chunks of the sequence. For each chunk one GEMM produces the logits and a kernel computes the loss in place. The gradient is computed in the same pass, so the backward pass only scales gradients that are already stored.

The chunk is sized so that the logits alive at any moment are about as large as the hidden states. For Qwen3-8B at 32,768 positions that is 1,024 rows per chunk and 32 chunks, with 311 MB of logits alive against 268 MB of hidden states (computed from the source of Liger 0.8.4).

The last column of the table is that loss. Its memory stays under 2.3 GiB at every length, most of it the gradient of the output matrix. In time it is 1.99 times slower than the plain loss at 4,096 positions, about equal at 16,384 and faster from there: 0.77 times at 32,768 and 0.71 times at 65,536.

Inside a whole training step the saving is in memory and the time does not move. On Qwen3-4B on one B300, with a forward and a backward pass per step, the step time with the fused loss stayed within 5% of the plain loss at every row length (measured):

| Row length | Gradient checkpointing | Peak, plain loss | Peak, fused loss |
|---|---|---|---|
| 16,384 | off | 133.8 GiB | 102.7 GiB |
| 32,768 | off | out of memory | 197.2 GiB |
| 16,384 | on | 43.1 GiB | 18.3 GiB |
| 32,768 | on | 78.7 GiB | 21.8 GiB |

With checkpointing on, the loss is most of what a step still holds, so the fused loss cuts the peak at 32,768 tokens by 3.6 times. Halo's benchmarks on other models recorded 133 to 54 GiB on Qwen3.5-9B at 65,536-token rows ([Liger Kernels → CrossEntropy vs FusedLinearCrossEntropy](../optimization/liger-kernels.md#crossentropy-vs-fusedlinearcrossentropy)).

The logits never exist as a tensor, so nothing that reads them can use this loss. A loss that needs the log-probability of each token or a whole distribution, as preference and distillation losses do, is one example, and logging the entropy of the output is another.

> In Halo: which training methods, parallel modes and model families use the fused loss, and which turn it off, is listed on [Liger Kernels](../optimization/liger-kernels.md).

#### What fusion does not help with

Small tensors gain nothing from the two kinds of fused kernel timed here. A Triton kernel is launched from Python, and a compiled function checks its guards, the conditions under which its compiled code is valid, before it runs. Either costs the CPU 25 to 31 µs a call, whatever the size. The same `silu(a) * b` was timed at four sizes (measured on one B300, back-to-back calls):

| Elements per tensor | Eager, 2 kernels | Liger's fused kernel | `torch.compile` |
|---|---|---|---|
| 4,194,304 | 10.2 µs | 30.8 µs | 24.8 µs |
| 16,777,216 | 28.7 µs | 30.5 µs | 25.4 µs |
| 67,108,864 | 115.9 µs | 59.0 µs | 63.5 µs |
| 268,435,456 | 442.1 µs | 229.7 µs | 245.3 µs |

At 4 million elements per tensor each fused kernel is slower than the two eager ones, whose two launches cost the CPU about 10 µs. At 16.8 million the three are close. From 67 million elements the fused kernels are 1.8 to 2.0 times faster. A fused kernel inside PyTorch itself pays the ordinary launch floor, so this threshold belongs to kernels launched from Python.

A single operation under an aggressive compile mode gets slower. `torch.compile(mode="max-autotune")` on one `silu(a) * b` takes 0.226 ms, 1.95 times the eager time and 3.5 times the time of the default compiled kernel. The fused kernel inside it is fine. The mode wraps the call in a CUDA graph and copies both inputs into fixed buffers on every call, which adds 7 copy kernels (measured on the B300).

Compilation has costs of its own. On Qwen3-30B-A3B with its experts on two GPUs, the first compiled step takes 6 to 15 s against 4 to 5 s eager. A recompile costs a step of 2 to 7 s, and peak memory rises by 9 to 23 GiB with Liger's kernels on ([torch.compile](../optimization/torch-compile.md#whole-model-compilation-opt-in)).

A compiler fuses only between the points where its graph breaks, the places where it has to hand control back to ordinary Python. A call into a hand-written kernel or into a communication library is such a point. On that same model at 16,384-token rows and batch 1, whole-model compilation gained at most 1% ([torch.compile → Graph breaks](../optimization/torch-compile.md#graph-breaks)).

A GEMM that already sits above the ridge gains little. Fusing a bias or an activation into its epilogue removes one read and one write of the output, a small share of the GEMM's time.

Check yourself:

1. A loop of 128 matmuls takes 1.12 ms. What does one matmul cost, and where does that time go?
2. `x = a * b + c` on three tensors of equal size: how many passes over memory does it take eager, and how many fused?
3. How large are the logits for 8,192 positions and a vocabulary of 200,000 entries, in bf16 and in fp32?
4. An update with momentum and no second moment reads the weight, the gradient and the momentum and writes the weight and the momentum, all in bf16. How long does it take for 8 × 10⁹ parameters at 6.65 TB/s?

[Answers](#answers-on-memory-bound-kernels-and-fusion) are at the end of the page.

### Attention

Attention as a textbook writes it has three steps. It computes a score for every pair of positions, `Q @ Kᵀ`, which is an `[S, S]` matrix per head for a row of `S` tokens. It takes a softmax over each row of the scores to get weights. It multiplies the weights by `V`. The letters `Q`, `K` and `V` are the queries, keys and values of the layer, and this `K` is not the GEMM width.

Written that way, the score matrix is the largest tensor in the model once the rows are long. At 32,768 tokens it has 1.07 × 10⁹ entries per head: 2.1 GB in bf16 and 4.3 GB in fp32 (computed). With 16 heads the fp32 scores alone are 68.7 GB, which is 64 GiB.

**Flash Attention** never stores the scores. It takes some of the queries and some of the keys, computes that tile of scores on the chip, folds it into the output and moves on to the next tile. For each query it carries two numbers along, a running maximum and a running sum, so it can finish the softmax without ever holding that query's whole row of scores. The running maximum is there because the exponential of a large score overflows. Subtracting the largest score seen so far keeps every exponent at or below zero. The backward pass computes each tile of scores again from the queries and keys, with one saved number per query to redo the softmax.

It is the same idea as the GEMM kernel: work on a tile that fits on the chip, and touch HBM once per tile.

#### What it changes and what it does not

The cost model is in the Flash Attention paper, and it is easy to misquote, so here it is in words.

- The FLOPs do not fall. The forward pass does the same arithmetic and the backward pass somewhat more, because it computes the scores again. They grow with the square of the row length.
- The extra memory falls from the square of the row length to a multiple of the row length. Long rows fit because of this.
- The traffic to and from HBM still grows with the square of the row length. It falls by a constant factor that grows with the size of the tile, and the paper reports up to 9 times less.

Put both forms on the roofline. The naive score GEMM has an AI near the head width. The whole naive operation also writes the scores and the softmax weights to HBM and reads them back, which brings its AI to about half of the head width in bf16. Head widths are 64 to 128 on many models and 256 or 512 on some, so on most models naive attention is below the ridge.

A Flash Attention kernel reads `Q`, `K` and `V` and writes the output, with neither of those two matrices between. If it read each input once, its AI would be half the row length. The real kernel reads `K` and `V` again for every set of queries it takes, so half the row length is an upper bound.

The three figures follow from one head of width `d` in bf16, forward pass, at a large `S` (computed):

```text
score GEMM            2×S²×d FLOPs over about 2×S² bytes of scores                    AI ≈ d
naive, both GEMMs     4×S²×d FLOPs over about 8×S² bytes, two matrices written, read  AI ≈ d/2
flash, inputs once    4×S²×d FLOPs over about 8×S×d bytes of Q, K, V and output       AI ≈ S/2
```

#### Measured by row length

The measurement is one causal attention call at batch 1, with 16 heads of width 128 in bf16, forward plus backward, on one B300. Naive is the three textbook steps as PyTorch operations. SDPA is PyTorch's `scaled_dot_product_attention`. FA2 and FA4 are two generations of Flash Attention kernel. Each cell is the time, then the rate, with the allocation above the inputs for the naive column.

The rates count the FLOPs of the causal half: `2 × S² × H` for the forward pass and 2.5 times that for the backward pass, with `H` = 16 × 128. That is the `7 × S × H` per token of [One training step](#one-training-step). The naive form multiplies the full matrices, so it executes about twice the FLOPs its rate credits.

| Row length | Naive | SDPA | FA4 | FA2 |
|---|---|---|---|---|
| 4,096 | 6.99 ms, 34 TFLOP/s, 4.05 GiB | 0.251 ms, 960 | 0.260 ms, 924 | 0.935 ms, 257 |
| 8,192 | 28.5 ms, 34, 16.1 GiB | 0.857 ms, 1,123 | 0.864 ms, 1,113 | 3.04 ms, 317 |
| 16,384 | 107 ms, 36, 64.4 GiB | 3.22 ms, 1,196 | 3.12 ms, 1,236 | 10.7 ms, 358 |
| 32,768 | out of memory | 12.5 ms, 1,235 | 11.8 ms, 1,301 | 40.5 ms, 381 |
| 65,536 | | 48.7 ms, 1,264 | 46.3 ms, 1,329 | 158 ms, 391 |

![Two log-log panels against row length from 1,024 to 65,536 tokens for one attention call, forward and backward, on a B300. Time: naive reaches 107 ms at 16,384 tokens and stops there, SDPA and FA4 lie together from 4,096 tokens up, at about 3 ms at 16,384, and FA2 is about 3.4 times slower. Peak memory: naive climbs to 64 GiB at 16,384 and is marked out of memory at 32,768 on the 267.7 GiB line. The three kernels share one line, 0.4 GiB at 16,384. An inset shows a score matrix with one tile lit.](../assets/diagrams/attention_by_length.png)

Four things are in this table.

1. The naive form runs out of memory at 32,768 tokens with the backward pass, on a GPU with 268 GiB. At 16,384 it holds 64.4 GiB, 4 times its fp32 scores. In the forward pass alone it still fits at 32,768 tokens, with 161 GiB, and runs out at 65,536.
2. The naive form runs at 20 to 36 TFLOP/s at every length, under 3% of the sustained GEMM peak. At 16,384 tokens it is 34 times slower than FA4. Most of that is traffic. The measured form makes the scores fp32 and spends a pass each on the scale, the mask, the softmax and the cast back, about 40 bytes for every score where the count above has 8 (counted from the benchmark's code). At about 13 FLOPs per byte the memory line allows about 85 TFLOP/s of executed arithmetic, about 43 by the count this table credits, and the forward pass alone measured 20 to 26.
3. SDPA and FA4 run near the sustained GEMM peak at long rows. In the forward pass alone they reach 1,493 to 1,506 TFLOP/s at 65,536 tokens. That is above the 1,388 of the GEMM loop because these kernels held the SM clock at 1,335 to 1,357 MHz under the power cap, where the GEMM loop held 1,185 to 1,260.
4. FA2 reaches 391 TFLOP/s, 29% of FA4. It computes the same attention on a kernel written for older tensor cores. FA2 ran at the full 2,032 MHz and FA4 under the power cap near 1,350 MHz, so at equal clocks the gap is larger.

The extra memory of SDPA and FA4 grows in proportion to the row length, to 1.51 GiB at 65,536 tokens with the backward pass. PyTorch's SDPA picked a kernel from cuDNN, NVIDIA's library of neural-network kernels, at every length here.

With 32 heads in place of 16, each call does twice the work, and from 16,384 tokens up the rates of SDPA, FA4 and FA2 stay within 4% of these. The naive form then holds 128.5 GiB at 16,384 tokens.

Short rows are overhead-bound. At 1,024 and at 2,048 tokens an FA4 forward call takes the same 0.058 to 0.059 ms, while its kernel alone takes 0.021 ms and 0.033 ms. At those lengths the CPU sets the time.

#### Three kernels for three generations

FA2, FA3 and FA4 compute the same attention on three generations of tensor core. FA2 targets the tensor cores of the Ampere generation. FA3 is written for Hopper's `wgmma` and TMA. FA4 is written for Blackwell's `tcgen05`, tensor memory and two-SM form, in a Python kernel language called CuTe DSL.

They are redesigns, each for its hardware. FA3 overlaps data movement with compute by warp specialization and interleaves the matmul with the softmax. FA4 exists because on Blackwell the tensor cores doubled in speed while shared memory and the exponential did not, so those two became most of the kernel's time. The papers report FA3 at 740 TFLOP/s on an H100, 75% of that GPU's peak, and FA4 at 1,613 TFLOP/s on a B200, 71%.

On a whole training step the gain you get from the newer kernel depends on the row length. In Halo's benchmark on one B300 (Qwen3-4B, batch 1), FA4 ran 1.19 times faster than FA2 at 4,096-token rows, where attention is a small share of the step, and 2.33 times faster at 32,768 ([Flash Attention](../optimization/flash-attention.md#fa4-vs-fa2-vs-sdpa-on-blackwell)).

SDPA and the Flash Attention kernels also differ in what they accept. The Flash Attention kernels handle head widths up to 256 and the cuDNN kernel behind SDPA up to 128 on this CUDA version, so if your model has wider heads, it cannot use that kernel. On a padded batch a variable-length Flash Attention kernel skips the pad tokens, and SDPA computes them and masks the result. Flash Attention kernels also cover model-specific attention variants that SDPA does not.

> In Halo: the attention kernel is chosen per GPU generation and per model family ([Flash Attention](../optimization/flash-attention.md)).

Check yourself:

1. How large is one head's score matrix at 16,384 tokens, in bf16? How large are the fp32 scores of 32 heads?
2. Flash Attention lowers two of these three and does not lower the third: FLOPs, memory, traffic to HBM. Which is which?
3. Naive attention with 32 heads held 128.5 GiB at 16,384 tokens in the measured run. How many copies of its fp32 scores is that?
4. By the count this section uses, how many FLOPs is one forward-and-backward call at 8,192 tokens with 16 heads of width 128? What rate does a time of 0.864 ms give?

[Answers](#answers-on-attention) are at the end of the page.

### Who writes the kernels

The kernels of the last sections come from different places. The list runs from the ones you call to the ones you write.

- cuBLAS is NVIDIA's closed matrix library. `a @ b` in PyTorch calls it, and through that path you get a plain GEMM chosen by shape. The library itself has more: grouped GEMM since version 12.5, fused epilogues in its cuBLASLt interface, and block-scaled fp4 and fp8 on Blackwell in 12.9. PyTorch's matmul does not expose those.
- `torch.compile` is a compiler. It fuses chains of PyTorch operations into Triton kernels it generates.
- Triton is the Python kernel language of the fused example. One Triton "program" is one block. You write loads, arithmetic and stores on blocks of values with masks, and the compiler maps them to threads. You choose the launch settings by hand: the block size, the number of warps and the number of buffer stages. The Liger kernels measured on this page are Triton kernels.
- CuTe DSL is a Python language for tensor-core kernels that comes with CUTLASS. It compiles a kernel at first use, and FA4 is written in it.
- CUTLASS is NVIDIA's open-source C++ template library for the kernel described in [Inside a tensor-core GEMM kernel](#inside-a-tensor-core-gemm-kernel). The author picks the tile, the stages, the scheduling and the epilogue. PyTorch's grouped GEMM is a CUTLASS kernel.

MoE training papers also cite DeepGEMM, DeepSeek's kernel library for fp8, fp4 and bf16 GEMMs, grouped ones for MoE among them, on Hopper and Blackwell.

How close does a higher-level language get on the tensor cores? PyTorch's compiler can generate a Triton GEMM from its templates and tune it, and on the B300 that kernel ran against cuBLAS on square bf16 operands (measured):

| GEMM | cuBLAS | Generated Triton kernel | Ratio |
|---|---|---|---|
| 4096³, both at 2,032 MHz | 1,690 to 1,698 TFLOP/s | 1,138 to 1,150 TFLOP/s | 0.68 |
| 8192³, from a cool start | 1,803 to 1,805 TFLOP/s | 1,214 to 1,215 TFLOP/s | 0.67 |
| 8192³, under the power cap | 1,389 to 1,435 TFLOP/s | 1,032 to 1,066 TFLOP/s | 0.74 |

At equal clocks the generated kernel reaches about two thirds of the library's rate. Under the power cap the gap narrows, probably because the Triton kernel keeps a higher clock. The GPU stood at 1,335 MHz after its run, where cuBLAS runs of this shape settle near 1,190 MHz. That is one reading, and the clock during the cuBLAS run of this comparison was not read.

Two published figures give the range. Triton's own tutorial for the Blackwell tensor core reports a hand-written matmul at 1,020 TFLOP/s without pipelining and 1,113 TFLOP/s with it, on a Blackwell GPU it does not name. The FA4 paper reports FA4 at 2.7 times a Triton attention kernel on a B200.

"Autotuning" means three different things, and it helps to ask which one a library means.

- A library that chooses among prebuilt kernels by shape when you call it. cuBLAS and cuDNN do this, and the sawtooth in the GEMM curve is the visible result.
- A compiler that times candidate kernels when it compiles. `torch.compile` in `max-autotune` mode does this.
- A person's sweep, frozen into a table of launch settings.

In all three, a tuned choice holds for one GPU architecture, one number format and one range of shapes.

> In Halo: one Triton kernel, the block-sparse attention kernel, keeps its tile, warps and stages in a table keyed by GPU generation. The others use one fixed setting ([Flash Attention → Block-sparse attention](../optimization/flash-attention.md#block-sparse-attention-qwen38-flash-next)).

### The first call is slow

The first time a process runs a kernel, the call can take thousands of times longer than the second. These are first, second and third calls in a fresh process, once with empty compile caches and once with the caches the first run filled (measured on one B300):

| Kernel | First call, empty caches | First call, filled caches | Second and third call |
|---|---|---|---|
| cuBLAS bf16 GEMM, 8192³ | 128 ms | 127 ms | 0.68 and 0.62 ms |
| a Triton kernel from Liger, 8192 × 8192 | 1,085 ms | 635 ms | 0.24 and 0.12 ms |
| `torch.compile` of `silu(a) * b` | 682 ms | 191 ms | about 0.2 and 0.12 ms |
| FA4 forward, 4,096 tokens, 16 heads | 2,446 ms | 2.9 ms | about 0.3 and 0.13 ms |

Three separate causes are behind these numbers.

Lazy loading is the first. The CUDA runtime loads a module of GPU code the first time it is needed. That is the default on Linux since CUDA 12.2 and was an option from 11.7. NVIDIA's guide warns that this "may skew performance measurements by moving CUDA module initialization into the measured execution window". The first cuBLAS call in a process shows it: 128 ms to create the library's handle and load its kernels, whatever the caches hold.

Just-in-time compilation is the second. Triton, CuTe DSL and `torch.compile` kernels are compiled when they are first used. The FA4 kernel takes 2.4 s to compile, and a filled cache brings its first call down to 2.9 ms. A filled Triton cache still leaves 0.6 s for the first launch of the Liger kernel, a time that was not broken down.

A new variant is the third, and it is the one that surprises people in the middle of a run. A compiled kernel is keyed on more than its code:

- Triton specializes a kernel on whether an integer argument is 1 or a multiple of 16, unless it is told not to.
- FA4 keys on the ratio of query heads to key heads, on a softcap value, which is a cap some models put on their attention scores, on a determinism flag and on whether a length arrives as a Python integer or as a tensor.
- `torch.compile` keys on batch size, gradient mode and types, and it stops recompiling after 8 variants.
- The cuBLAS library picks a kernel by the shape of the operands.

A step whose shape or setting has not been seen before compiles or loads again.

These causes lead to three habits.

1. Run every kernel variant once at load, before the steps you time.
2. Keep the shapes of a run inside the set you warmed up.
3. Keep the compile cache on a local disk.

The second habit removed the slow steps of one 8-GPU run, recorded on Halo 1.2.0. Its optimizer kernel updates many tensors in one launch and takes them as a table. With a table of fixed size no new variant of the kernel is compiled during the run, and the slowest step stayed at 1.01 times the median step. With a table that grows during the run it was 1.44 to 1.48 times ([BF16 Optimizer](../optimization/bf16-optimizer.md#benchmarks)).

The third habit matters at scale because every process compiles the same kernels. Each uses up to 32 compiler threads, so an 8-GPU node starts up to 256 of them, and a cache directory on a shared network filesystem makes them queue on file locks ([torch.compile → Compile caches at scale](../optimization/torch-compile.md#compile-caches-at-scale)).

With several GPUs a compile on one of them delays all of them. The section on how distributed runs fail explains why.

### Which limit a kernel is hitting

You have now met three limits on the speed of one GPU: the arithmetic rate, the rate of memory traffic, and the cost that each kernel carries whatever its size. For a loop of one kind of work you can tell them apart from outside the program, before you open a profiler.

#### Power and utilization

`nvidia-smi` defines `utilization.gpu` as the "percent of time over the past sample period during which one or more kernels was executing on the GPU". A kernel that is waiting for memory counts as executing. So does a kernel that is waiting for another GPU. The figure says that something was running and nothing about what it did.

Board power and the SM clock say more. Each row here is a loop of one kind of work, sampled for 20 s (measured on one B300, with a second one for the last row):

| What the GPU is doing | Board power | Share of the 1,100 W limit | Utilization | SM clock | Rate |
|---|---|---|---|---|---|
| no process | 185 W | 17% | 0% | 120 MHz | |
| a process that holds the GPU and does no work | 237 W | 22% | 0% | 2,032 MHz | |
| large bf16 GEMMs, 8192³ | 1,092 W | 99% | 100% | 1,187 MHz, power cap | 1,384 TFLOP/s |
| elementwise `silu(a) * b`, eager | 1,029 W | 94% | 100% | 2,032 MHz | 5.78 TB/s |
| elementwise `torch.addcmul` on 4 GiB tensors, 3 reads and 1 write | 875 W | 80% | 100% | 2,032 MHz | 7.24 TB/s |
| small GEMMs, `M = 128`, `K = N = 2880` | 666 W | 61% | 93% | 2,032 MHz | 246 TFLOP/s in this run, 237 in the sweep |
| a loop over 128 experts, 256 small matmuls | 374 W | 34% | 59% | 2,032 MHz | 64 TFLOP/s |
| 1-element kernels | 241 W | 22% | 52% | 2,032 MHz | 200,735 kernels/s |
| a whole training step, Qwen3-4B on one row of 8,192 tokens | 1,042 W | 95% | 98% | 1,927 MHz on average, power cap at times | 381.6 ms per step |
| waiting for another GPU inside a communication call | 238 W | 22% | 100% | 2,032 MHz | |

The last row was measured on two B300s. One process entered a communication call that needs both GPUs, and the other joined it 30 s later. The part on many GPUs explains such calls.

![Bars for ten states of one B300 in order of board power, in three columns: power against the 1,100 W limit, utilization, and SM clock. Four states read 100% utilization: a rank waiting in a collective at 238 W, two elementwise loops at 875 and 1,029 W and a large GEMM loop at 1,092 W. The GEMM loop runs at 1,187 MHz and a whole training step at 1,042 W, 98% and 1,927 MHz, both under the power cap. A second panel: the GEMM loop at 1,790 TFLOP/s for 39 ms, then 1,392.](../assets/diagrams/power_by_regime.png)

The second panel is the first second of the GEMM loop, the burst and the fall that [The chip](#the-arithmetic-rate) described. It reads 1,392 TFLOP/s after the burst, and over the whole 20 s the loop averages the 1,384 of the table.

The usual advice is to watch power, not utilization. The table supports it and adds the clock. For a loop of one kind of work on this board:

- Utilization at 100%, power at the limit and a throttled SM clock is compute-bound work. It is the only loop in the table whose clock is under 2,032 MHz.
- Utilization at 100% with the clock at its maximum and power far above idle is memory-bound work. Its power is 80 to 94% of the limit, so power alone does not separate it from the compute-bound row. The clock does, because on this board a loop of large GEMMs runs into the power limit.
- Utilization near half, with power at 22 to 34% of the limit, is overhead-bound work.
- Utilization at 100% with power at the level of an idle process, 22% of the limit, is a GPU that waits. A loop of tiny kernels draws the same power and reads about half.
- Utilization at 0% with the power of an idle process is a GPU with no kernel queued. The CPU side of that rank is doing something else, or nothing.

A loop of small GEMMs sits between the second and third case, at 93% and 61%. To tell it from a memory-bound loop you need a profile.

A whole training step is a mix of all of these. The measured one, a full fine-tuning step of Qwen3-4B, draws 95% of the limit at 98% utilization. Its clock moves between 1,665 and 2,025 MHz, and the driver reports the power cap during parts of the step. By the first reading it looks compute-bound, and yet fused kernels make the same model's step a quarter to a third shorter.

On a real step the readings can show two cases when they fill the step: utilization near half at a quarter of the power limit is a step paced by the CPU, and 100% at the power of an idle process is a rank that only waits. A step that does either for a third of its time reads high on both, so a high reading rules nothing out. Comparing the ranks and reading a profile do.

Sample these fields together during a steady step. The last one says whether the driver is holding the clock down for power:

```bash
nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,clocks.current.sm,clocks_event_reasons.sw_power_cap --format=csv -l 1
```

#### The limits side by side

| The step is | What limits it | What to change |
|---|---|---|
| compute-bound | the tensor cores, at the clock the power cap allows | a lower precision where [Precision](#precision) says it helps; a kernel that does the same FLOPs for fewer watts |
| memory-bound | the bytes moved per FLOP | a larger `M` for a GEMM, fusion for elementwise work, Flash Attention for attention |
| overhead-bound | the cost of each kernel that does not shrink with its work, on the CPU and on the GPU | fewer and larger kernels, and less CPU work per kernel |

The overhead row covers two costs. The CPU needs about 5 to 12 µs to launch a kernel, depending on the operation and the process. And a small GEMM takes about 2.5 to 5 µs on the device at small `M`, a time that rises with `K`, plus a share that grows with the bytes of its weight. The table has no row for "too few tiles to fill the GPU": with cuBLAS on a B300 the fine sweep of `M` showed no such regime.

Check yourself:

1. A loop of one kind of work on this board reads 100% utilization, 900 W and an SM clock of 2,032 MHz. Which limit is it, and what do you change?
2. Another loop reads 55% utilization, 300 W and 2,032 MHz. Which limit is it, and what do you look for in a profile?
3. A third reads 100% utilization, 1,095 W and 1,200 MHz, and it is still slower than you need. What is left to change?
4. One GPU of an 8-GPU job reads 100% utilization, 238 W and 2,032 MHz. What is that GPU doing?

[Answers](#answers-on-which-limit-a-kernel-is-hitting) are at the end of the page.

### Precision

The limits so far are on speed. The next three sections are about what else a run has to get right: what it computes accurately, what it computes the same way twice, and what fits in memory.

A floating-point format divides its bits between an exponent, which sets the range of values, and a mantissa, which sets how finely they are spaced. Training stores most of its numbers in 16 bits. Sixteen bits are enough in most places and too few in some, and a run has to treat those places differently.

#### Number formats

| Format | Exponent and mantissa bits | Largest value | Decimal digits | Spacing just above 1.0 |
|---|---|---|---|---|
| fp32 | 8 and 23 | 3.4 × 10³⁸ | about 7 | 1.2 × 10⁻⁷ |
| fp16 | 5 and 10 | 65,504 | about 3 | 9.8 × 10⁻⁴ |
| bf16 | 8 and 7 | 3.4 × 10³⁸ | about 2 | 7.8 × 10⁻³ |

bf16 keeps the exponent of fp32 and gives up mantissa. Its relative precision is 2⁻⁷, about 0.8%. fp16 keeps more mantissa and has a 5-bit exponent, which fails at both ends. Values above 65,504 overflow, and small gradients underflow: PyTorch's notes on mixed precision say of them that "these values will flush to zero". The technique called loss scaling exists to work around the underflow. bf16 has the range of fp32, so it needs none.

[The chip](#the-chip) gave the two speeds of an fp32 matmul: a twentieth of bf16 at full precision, and half of bf16 with TF32. TF32 gets its speed by rounding the inputs to 10 mantissa bits. The PyTorch setting is `torch.set_float32_matmul_precision`, where `highest` is full precision and `high` allows TF32.

**TF32 can arrive without being asked for.** NVIDIA's base container image turns it on. Ten stored mantissa bits give 11 significant bits, so integers are exact only up to 2¹¹ = 2,048 (computed). A rotary position encoding that passes position numbers through an fp32 matmul then gives several positions past 2,048 the same value.

The words "mixed precision" name two setups. In the first, called autocast, the parameters stay in fp32 and the inputs of each operation are cast down as it runs. In the second, the parameters themselves are stored in bf16. The second halves the memory per parameter, as the next subsection works out.

> In Halo: parameters are stored in bf16, and the fp32 matmul precision is pinned to `highest` when a model is loaded ([Configuration Reference → Environment variables](configuration-reference.md#environment-variables)).

#### What stays fp32 in a bf16 run

Two digits are enough for a weight or an activation. They are not enough for a long sum, and not for a small update to a weight. A bf16 run handles those two cases in fp32.

Take the sums first. Add many bf16 numbers into a bf16 total, and once the total is 256 to 512 times the next term, that term is at most half a step of the total and is dropped.

The measurement on the B300 multiplies pairs of numbers between 0 and 1 and adds the products into a bf16 total 256 at a time, so each term it adds is about 64. The total stops growing near 32,768. After 262,144 products it holds half of the true sum, and after 16.7 million it holds 0.8%. By the same rule a total built one product at a time would stall after a few hundred terms (not measured).

So everything that adds many numbers does it in fp32:

- The tensor cores do, with the fp32 accumulator that [Inside a tensor-core GEMM kernel](#the-accumulator-is-fp32) showed.
- PyTorch's `sum` on a bf16 tensor does. It stays within 10⁻⁷ of the exact value when the result is read out in fp32 (measured on the B300, up to 16.7 million terms).
- The normalization layers of the models measured here normalize in fp32 and cast back.
- The loss converts the logits to fp32.
- The attention kernels run the softmax and add up the output in fp32.

One sum is the exception. The reduction of gradients across GPUs is done in bf16 unless you ask for fp32, and its error grows with the number of GPUs.

Now the weight update. Take a weight near 0.02. The value 0.02 itself is not a bf16 number, and the nearest one is 0.02001953125. It lies between 2⁻⁶ and 2⁻⁵, where neighboring bf16 values are 2⁻¹³ = 1.22 × 10⁻⁴ apart, 64 times closer than just above 1.0. That gap is called the **ULP**, the unit in the last place. An update of 10⁻⁵ is less than half of it. Round-to-nearest puts the weight back where it was, and it does so on every step, so the weight never moves however long you train.

![Three lanes show the representable values around 0.02. fp32 is a solid band, fp16 has 27 ticks 1.53 × 10⁻⁵ apart, and bf16 has four ticks 1.22 × 10⁻⁴ apart. An update of 10⁻⁵ from the bf16 weight lands 8% of the way to the next value. Round to nearest returns it. Stochastic rounding moves it on with probability 8%.](../assets/diagrams/bf16_number_line.png)

**Stochastic rounding** fixes this without more bits. It rounds up with a probability equal to the fraction that would be dropped. Here the weight moves one ULP with probability 10⁻⁵ / 1.22 × 10⁻⁴ = 8%, and stays otherwise. Averaged over steps, it moves by the update that was asked for.

The optimizer has state of its own to round. AdamW keeps two running averages per parameter: the **first moment**, an average of the gradient, and the **second moment**, an average of its square. The second moment is never negative, and round-to-nearest biases its running value by more than 15%, so it needs stochastic rounding too.

The older remedy is **master weights**: an fp32 copy of every weight and of both moments, cast down to bf16 for the forward and backward passes. It costs 12 bytes per parameter, and the bf16 working copy adds 2 more. Weights and moments stored in bf16 cost 6. For 20 × 10⁹ parameters that is 240 GB of fp32 state, before the working copy, against 120 GB.

How well does the 6-byte version track the 12-byte one? On a 4-layer test model of 50M parameters, the loss after 200 steps is within about 8 × 10⁻⁵ of the fp32-master run. With stochastic rounding on the weight only, it is within about 4 × 10⁻⁴. Plain bf16 with round-to-nearest lags by 0.27 (recorded, [BF16 Optimizer](../optimization/bf16-optimizer.md#benchmarks)).

One step does not show the difference. The measurement is a single optimizer step on Qwen3-8B at a learning rate of 10⁻⁴, with random gradients, on one B300. After it, 21.0% of the weights of the first layer had not moved under round-to-nearest, and 18.6% under stochastic rounding. The two differ in what happens next. By the rule above, round-to-nearest leaves the same weights in place on every step, and under stochastic rounding every weight has its chance on every step. Only this one step, at one learning rate, was measured.

So a "bf16 run" means bf16 storage, fp32 accumulation in the tensor cores, fp32 sums in the norms, the softmax and the loss, and an optimizer that rounds stochastically.

> In Halo: the default optimizer, AdamWBF16, computes each update in fp32 in registers and rounds only when it stores. It uses stochastic rounding on the weight and the second moment and round-to-nearest on the first moment ([BF16 Optimizer → Stochastic rounding](../optimization/bf16-optimizer.md#stochastic-rounding)). An option reduces gradients across GPUs in fp32. On 8 GPUs that is about 2.2 times tighter, 0.17% against 0.37% of the signal, for about twice the bytes of that transfer ([Master weight and grad-reduce options](../optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options)).

#### When low precision speeds training up

Formats below 16 bits raise the peak of the tensor cores. Halo's registry holds 2,250, 4,500 and 9,000 TFLOP/s for bf16, fp8 and fp4 on a B300, and its fp4 figure keeps the B200's ratio of 4 times bf16. NVIDIA's own table for the HGX B300 lists dense fp4 at 108 × 10¹⁵ FLOP/s for 8 GPUs, which is 13,500 TFLOP/s per GPU. fp4 tensor cores exist on Blackwell only.

Low-precision formats keep small values alive with **block scaling**. They store one scale per small block of values, so a few large values do not push the rest to zero. The MX formats (mxfp8, mxfp4) use one power-of-two scale per 32 values. nvfp4 uses one finer scale per 16 values and a global scale per tensor.

The kernels deliver the higher peak on a large square GEMM. The measurement is an 8192³ GEMM on one B300, with the operands quantized ahead of time and every arm run long enough to sit at the power cap. It ran at 1,404 TFLOP/s in bf16, 2,644 in fp8, 2,608 in mxfp8 and 5,565 in nvfp4. Those are 1.9 times and 4.0 times bf16 under the same 1,100 W.

![Bars of the rate of an 8192³ GEMM on one B300 in two groups, with a tick for the rate in the first 40 ms. Stored formats: bf16 1,432 TFLOP/s in back-to-back calls and 1,793 at first, fp16 1,346 and 1,694, fp32 with TF32 761 and 955, full-precision fp32 68. Quantized operands: bf16 1,404, fp8 2,644, mxfp8 2,608 and nvfp4 5,565, which is 4.0 times bf16.](../assets/diagrams/precision_ceilings.png)

The upper group of the figure repeats the formats of [The chip](#the-arithmetic-rate) from another run, with samples of about 60 ms, in which the sustained bf16 rate read 1,432 TFLOP/s.

On narrower weights the gain shrinks. With 16,384 rows and widths of 2,048 and 2,880, fp8 ran 1.4 to 1.5 times bf16 and nvfp4 1.4 to 2.1 times. Those runs were shorter and their arms did not all sit at the same power state, so the ratios mix the kernel with the clock. Accuracy falls with the bits: against the bf16 result, the fp8 output was off by 3.8% and the nvfp4 output by 13.4% on random operands.

The roofline says when a higher peak helps. A compute-bound GEMM runs at the peak, so a higher peak makes it faster. A memory-bound GEMM does not reach the peak. Most low-precision schemes also add a pass that reads the operand, computes the scales and writes a quantized copy, so a memory-bound GEMM gets slower.

Whether an expert GEMM is above its crossover M depends on its shape. For a weight that is not square the crossover M is `r×K×N / (K×N − r×(K + N))`. At a ridge of 209 that is 244 rows for `K = N = 2880`, 301 rows for `K = 2048, N = 1024` and 427 rows for `K = 512, N = 2048` (computed). Narrow experts stay near their flat small-`M` time well past those row counts: a 2048 × 1024 GEMM takes 4.2 µs on the device at 128 rows and 4.6 µs at 512.

A kernel that is faster alone can still make the step slower. Halo's measurements show how it happens ([Low-Precision MoE Kernels → Why bf16 is the default](../optimization/low-precision-moe-kernels.md#why-bf16-is-the-default)):

- The cast costs more than the kernel saves. On the 1.2.0 development build's adapter, at one gpt-oss-20b expert shape, casting the activations took 2.66 ms where the whole bf16 forward GEMM took 0.86 to 0.89 ms.
- Some grouped low-precision kernels need the rows of each expert laid out in a fixed pattern. Routing changes that pattern at every step, so the layout is rebuilt on the CPU at every call.
- Whole steps on native fp8 and fp4 kernels came out slower on the 1.2.0 development build's adapter. On 8 B300s with experts over all 8 and 4,096-token rows they ran at 0.37 to 0.54 times the bf16 rate. The expert GEMMs of those runs were far above their crossover M, at 16,384 rows per expert on gpt-oss-20b, so the roofline allowed a gain and the casts took it back.
- A simulated path, which quantizes, dequantizes and then runs the bf16 matmul, executes no low-precision instruction and cannot beat bf16. It ran at 0.10 to 0.45 times the bf16 rate on the B300s, and at 0.93 to 0.99 times on 8 H200s, where the round trip of the activations is fused into one kernel.
- A quantized weight against a bf16 activation gains no compute speed on Blackwell, because the tensor core has no instruction that multiplies a bf16 operand by an fp4 one.

Large runs elsewhere do report gains, and the scope of each figure matters. DeepSeek-V3, a model of 671B parameters with 37B active per token, ran all three GEMMs of its linear layers in fp8 on 2,048 H800 GPUs. Its paper says this "theoretically doubles the computational speed", a statement about the kernels. It kept the embedding, the output head, the router, the norms and attention in higher precision. It also kept master weights and optimizer state in higher precision, and it reports a loss within 0.25% of the higher-precision run.

For whole steps, other stacks have reported 1.13 to 1.22 times for mxfp8 and 1.34 to 1.37 times for nvfp4. Halo's page names each of those runs ([Low-Precision MoE Kernels → Formats](../optimization/low-precision-moe-kernels.md#formats--mx-vs-nv)).

When you read such a number, check whether it is for one kernel or for a whole step.

> In Halo: bf16 is the default. The low-precision paths are for quantization-aware training and for exporting a quantized checkpoint ([Low-Precision MoE Kernels](../optimization/low-precision-moe-kernels.md)).

Check yourself:

1. A bf16 weight is near 0.5 and the update is 10⁻⁴. What does round-to-nearest do? What does stochastic rounding do?
2. Up to which integer is fp16 exact, and why is that the same answer as for TF32?
3. A model has 8B parameters. How much do weights and two moments take in bf16, and how much do fp32 master weights and moments take?

[Answers](#answers-on-precision) are at the end of the page.

### Determinism

Run the same step twice with the same seed and the two losses can differ in their last digits. Floating-point addition is not associative, so a kernel whose order of summation depends on timing gives slightly different bits from run to run. PyTorch's notes on reproducibility say that "completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms", and that "deterministic operations are often slower than nondeterministic operations".

Three sources are usual. Many threads add into one memory location in whatever order they arrive. A parallel sum has a shape that depends on launch settings. Or a library is free to choose a different algorithm on each run.

The first source is slow as well as unordered. A kernel that writes many values into positions chosen by an index is called a scatter. When several values land on one position, each thread has to add its value without losing another thread's, an operation called an atomic add. The bf16 `index_add_` kernel has no native atomic add. It emulates one with a loop that reads the value, adds to it and tries again if another thread got there first, and that loop makes the threads take turns when many rows hit the same index.

On one profiled gpt-oss-20b step with its experts split across GPUs that kernel was about 20% of all GPU time, and it lengthened the wait of the other ranks as well. The path that replaced it ran about 5 times faster than the default backward pass. On packed chat data, replacing the same kind of scatter where tokens are put back in order gained 24% on gpt-oss-20b with its experts over 8 GPUs ([Grouped GEMM → The duplicate-index gather trap](../optimization/grouped-gemm.md#the-duplicate-index-gather-trap)).

The replacement is a gather, in which every output position reads the values it needs, followed by a reduction. If you find a scatter with repeated indices in your own profile, the same move makes it both deterministic and fast.

In bf16 a reordered sum can change a discrete choice, and the two runs then differ by more than their last digits. An MoE router picks the top `k` experts by score. When two scores are close to a tie, adding the sums behind them in another order can swap them, and the token goes to a different expert.

With several GPUs there are two more sources: the order in which tokens arrive from other GPUs, and any change in how a sum is split between GPUs.

Determinism costs speed. On gpt-oss-20b on 2 B300s at 4,096 tokens per GPU, a deterministic version of the transfer that carries tokens to experts on other GPUs ran at about 14,900 tokens/s per GPU against 15,100, between 1 and 2% slower. The full deterministic mode ran at about 13,400, about 11% slower, and most of that is PyTorch's deterministic kernels ([Expert Parallelism → Determinism](../parallelism/expert-parallelism.md#determinism)).

A determinism flag is also one of the variant keys listed in [The first call is slow](#the-first-call-is-slow), so turning it on after warm-up forces a kernel to compile again.

> In Halo: splitting attention across GPUs reorders the sums the MoE router reads and flips a few near-tied expert choices. On gpt-oss-20b that shows as a loss offset of 0.060, and it is not a bug ([Tensor Parallelism → EP+TP mode](../parallelism/tensor-parallelism.md#eptp-mode)).

### Memory

The limit here is capacity. Everything a step needs has to fit in HBM at once, and when it does not, the step fails with an out-of-memory error.

#### What fills HBM

Three of the four things in HBM scale with the number of parameters. In a bf16 run:

| What | Bytes per parameter | Note |
|---|---|---|
| weights | 2 | |
| gradients | 2 | exist during a step, and only for trainable parameters |
| optimizer state | 4 with bf16 moments, 8 with fp32 moments | AdamW keeps two moments |
| activations | not per parameter | they scale with the tokens in the step |

So 6 bytes per parameter persist from step to step, for the weights and two moments. During a step the gradient makes it 8. With fp32 master weights and moments the state is 12, and the bf16 working copy adds 2. A frozen parameter costs 2. The lower half of the figure in [One training step](#one-training-step) draws these bars.

Check the count against a measurement. gpt-oss-20b has 20.9B parameters, and in one benchmark about 14B of them were trainable. The weights are 20.9 × 2 = 41.8 GB. Gradients and moments for the trainable part are 14 × (2 + 4) = 84 GB. The sum is 125.8 GB (computed). The recorded peak on one B300 is 134.2 GB ([BF16 Optimizer](../optimization/bf16-optimizer.md#benchmarks)), which leaves about 8 GB for activations and everything else.

Splitting a model over several GPUs divides some of these four and leaves others whole. The part on many GPUs says which, mode by mode.

#### Activations

**Activations** are the intermediate tensors of the forward pass that the backward pass needs. Autograd keeps them from the moment the forward pass produces them until the backward pass has used them. They grow with batch size × row length × hidden size × number of layers.

Two properties set them apart from the other three. They do not shrink when the parameters are split across GPUs. And parameters that a token does not use add nothing to them: the unused experts of an MoE layer are an example.

One bf16 hidden state at 32,768 tokens and a hidden size of 4,096 is 32,768 × 4,096 × 2 = 268 MB (computed). A decoder layer keeps several of them, and the intermediates of its MLP are wider than the hidden size.

How many is "several"? A measurement on Qwen3-4B on one B300 gives the number for one model. It has 4.02B parameters, 36 layers and a hidden size of 2,560, and it ran a forward and a backward pass with the plain loss and no gradient checkpointing:

| Rows × tokens per row | Step time | Held after the forward pass | Peak memory |
|---|---|---|---|
| 1 × 4,096 | 178.5 ms | 26.94 GiB | 39.09 GiB |
| 1 × 16,384 | 813.9 ms | 107.77 GiB | 133.82 GiB |
| 4 × 4,096 | 670.5 ms | 107.76 GiB | 133.82 GiB |
| 16 × 1,024 | 631.4 ms | 107.76 GiB | 133.82 GiB |
| 1 × 32,768 | out of memory | | |

The forward pass leaves 6.73 MiB per token for the backward pass to use. Of that, 5.87 MiB is in the decoder layers, which is 167 KiB per token per layer. One bf16 hidden state of this model is 5 KiB per token, so a layer keeps about 33 hidden states' worth (computed). The rest is the loss. Beside the activations, the weights and gradients of Qwen3-4B are a fixed 15 GiB.

The same measurement on Qwen3-8B, with a hidden size of 4,096, gives 8.29 MiB per token, of which 211 KiB per token per layer is in the decoder, about 26 hidden states' worth.

The table shows two more things. Activations depend on the number of tokens and not on how the tokens are arranged: 16,384 tokens hold the same 107.8 GiB as one row, as 4 rows or as 16. The step time does depend on the arrangement, because the cost of attention grows with the square of each row's length.

Your model's weights can fit and the step can still run out of memory at long rows. Four things lower activation memory, each by a different factor. Gradient checkpointing removes the factor of layers. Flash Attention removes the term that grows with the square of the row length. The fused loss removes the logits. And two of the parallel modes in the next part divide the activations a GPU holds.

#### Allocated and reserved

PyTorch does not return freed GPU memory to the driver. It keeps it in pools for reuse. That gives you three different numbers for "memory used":

- `torch.cuda.memory_allocated()` reports the memory that tensors occupy now.
- `torch.cuda.memory_reserved()` reports the memory the allocator holds, used or not.
- `nvidia-smi` shows the reserved amount plus the memory the CUDA runtime itself needs.

Allocated is at most reserved, and reserved is at most what `nvidia-smi` shows.

**Fragmentation** is reserved memory that is free and has the wrong size. A request fails while the error message reports gigabytes as "reserved but unallocated". It comes from shapes that change from step to step. The allocator rounds each request up, cuts blocks to size and keeps separate pools for small and large requests, so a run of changing sizes leaves pieces that fit nothing.

The allocator option `expandable_segments`, set through the environment variable `PYTORCH_CUDA_ALLOC_CONF`, addresses it. On a dense packed run it lowered reserved memory from 41.3 to 35.4 GB with no change in speed. With experts split across GPUs it saved 1 to 6 GB and cost about 1.5% in speed ([Throughput Benchmarks → Allocator](../optimization/throughput-benchmarks.md#allocator-expandable-segments)).

The pools are per stream, which matters once a library puts work on a second stream, as communication code does. The allocator's source says: "Once freed, blocks can be re-allocated on the same stream, but not on any other stream". Inside one stream the kernels run in order, so a freed block is safe to hand out again. Another stream has no such order with the kernels that may still be using the block.

A block freed on a stream that is not used again stays in that stream's pool. It is released when an allocation fails and the allocator frees its cached blocks to retry, or when `torch.cuda.empty_cache()` runs. If one GPU of a job shows reserved memory well above its peak allocated memory, look for pools stranded this way ([Debugging → GPU memory profiling](debugging.md#3-gpu-memory-profiling)).

> In Halo: the buffers of the RL weight transfer are allocated on the caller's stream for this reason ([Rollout Servers → Streaming and quiesce](../infrastructure/rollout-servers.md#streaming-and-quiesce)).

#### Gradient checkpointing

**Gradient checkpointing** saves the input of each decoder layer and discards everything the layer computed inside. When the backward pass reaches a layer, it runs that layer's forward a second time to rebuild what it needs. One input is kept per checkpointed layer. A setting that checkpoints only every n-th layer keeps one input for each of those and all the activations of the layers between.

The second forward adds `2 × P` FLOPs per token to the `6 × P` of a normal step, a third more arithmetic. The measured cost in time is a little lower, 22 to 32% across Halo's benchmark runs, because part of a step is not forward arithmetic. In one of those runs the step took 29% longer, 13,239 against 17,018 tokens/s per GPU. That run is gpt-oss-20b on 8 B300s with its experts spread over all 8, batch 4 and 4,096-token rows (recorded, [Throughput Benchmarks](../optimization/throughput-benchmarks.md#where-the-ep-steps-time-goes)).

In memory, turning checkpointing off costs 1.4 to 2.0 times the peak memory on those same runs.

The same Qwen3-4B measurement as in [Activations](#activations) shows both sides on one GPU (forward and backward pass, plain loss):

| Rows × tokens per row | Step time, off and on | Held after the forward pass, off and on | Peak, off and on |
|---|---|---|---|
| 1 × 4,096 | 178.5 and 231.2 ms | 26.94 and 4.26 GiB | 39.09 and 17.62 GiB |
| 1 × 16,384 | 813.9 and 1,045.5 ms | 107.77 and 17.04 GiB | 133.82 and 43.10 GiB |
| 1 × 32,768 | out of memory and 2,599.6 ms | 34.09 GiB with it on | 78.69 GiB with it on |

From 4,096 tokens up, checkpointing costs 28 to 30% in step time, and the backward pass grows by the forward pass it repeats. What it keeps of the decoder is about one input per layer: 0.20 GiB per 1,024 tokens for the 36 layers, where one input per layer comes to 0.18 GiB (computed). Most of what is still held with checkpointing on is the loss: 13.9 of 17.0 GiB at 16,384 tokens.

![Stacked bars of allocated memory at three moments of one step of Qwen3-4B on a B300 at 16,384 tokens, without and with gradient checkpointing. Before the step both hold 7.5 GiB of weights. After the forward: 115 GiB without checkpointing, 93.9 of it saved by the decoder stack and 13.9 by the output head and the loss, and 24.6 GiB with it. At the peak the backward has added 18.5 GiB: 134 and 43.1 GiB.](../assets/diagrams/memory_over_step.png)

The figure stacks the 7.5 GiB of weights under what the forward pass saves, so its bars after the forward pass read 115 and 24.6 GiB where the table has 107.8 and 17.0.

The rule that follows has two halves, and people tend to remember only the first. At equal batch, turning checkpointing off is faster whenever the batch still fits. But checkpointing also lets a larger batch fit, and the largest batch with checkpointing can beat the largest batch without it. Compare the two before you decide.

> In Halo: gradient checkpointing is on by default. An opt-in plan callback, for packed supervised rows, measures the second step and applies the lightest checkpointing that fits. It ran 25.5% faster than full checkpointing on one packed dense run ([Callbacks → GradientCheckpointingPlanCallback](../training-methods/callbacks.md#gradientcheckpointingplancallback)).

PyTorch's `checkpoint` function, which implements gradient checkpointing, comes in two kinds. The **reentrant** kind runs the layer under `no_grad` in the forward pass, so nothing is recorded. In the backward pass it runs the layer again with gradients on and calls autograd from inside autograd, and the name comes from that. Its output carries a gradient only if one of its inputs does.

The non-reentrant kind records the graph in the forward pass without keeping the tensors, and computes them again the first time the backward pass asks for one. Which of the two a run can use depends on its parallel mode.

Megatron-LM, NVIDIA's training system for large language models, has a lighter variant. Its **selective recomputation** checkpoints only the parts of a layer that hold the most memory, the attention operations. Its paper reports 70% and 65% of activation memory saved on two models for 2.7% and 1.6% more FLOPs, where full recomputation costs 30 to 40% more time. With Flash Attention little of that memory is left to save, and checkpointing whole layers remains.

> In Halo: expert and context parallelism force the reentrant kind, and a pipeline needs the non-reentrant kind ([Pipeline Parallelism → Rejected combinations](../parallelism/pipeline-parallelism.md#rejected-combinations)). Under expert parallelism the recompute replays the cached transfer of tokens to their experts ([Expert Parallelism → Gradient-checkpoint dispatch replay](../parallelism/expert-parallelism.md#gradient-checkpoint-dispatch-replay)).

#### Optimizer state

The two moments of AdamW are 4 bytes per parameter in bf16 and 8 in fp32. Three kinds of optimizer change that count, each for a different purpose.

- bf16 moments with stochastic rounding keep the state at 4 bytes. Fused into one kernel, this is also the fastest update of the three, the one measured in [Predicting a memory-bound kernel](#predicting-a-memory-bound-kernel).
- 8-bit moments go lower, with the block-wise dynamic quantization of the bitsandbytes 8-bit optimizers (arXiv 2110.02861). With a 24-bit master weight, everything comes to about 5 bytes per parameter, against 6 and 12, and the step gets longer.
- An optimizer that keeps one moment has 2 bytes of state. Muon is one. It changes the update itself, and its smaller state does not make it lighter in peak memory or in time.

> In Halo: the three are AdamWBF16, which is the default ([BF16 Optimizer](../optimization/bf16-optimizer.md)), FlashAdamW and Muon. FlashAdamW took about 50 s against 34 s per step on one long-row run, GLM-4.7-Flash with its experts over 8 GPUs and 32,768-token rows ([FlashAdamW](../optimization/flash-adamw.md)). On an all-matrix micro-benchmark that its page calls a worst case, Muon's peak memory was 68% higher and its step took 82.4 ms against 7.7 ms ([Muon](../optimization/muon-optimizer.md#benchmark)).

#### Frozen weights and LoRA

A frozen weight costs `4 × P` FLOPs per token and 2 bytes per parameter. A trained one costs `6 × P` and 8 bytes. By that count, freezing saves at most a third of the FLOPs of the linear layers. The memory it saves is in gradients and optimizer state. Activations shrink by less than you might expect, because the backward pass still runs through every layer above the lowest trainable parameter.

**LoRA** is the common case. It freezes the model and trains small added matrices beside chosen weights.

Qwen3-4B ran whole training steps on one B300, on one row of 4,096 tokens with gradient checkpointing off, under four choices of what is trained (measured). Between them the rows show the three cases of the `6 × P` rule: a trained layer, a frozen layer the gradient passes through, and a frozen layer below the lowest trained one.

| What is trained | Trainable parameters | Forward | Backward | Step | Held after the forward pass | Peak memory |
|---|---|---|---|---|---|---|
| everything | 4,022.5M | 63.1 ms | 116.1 ms | 188.8 ms | 26.95 GiB | 54.09 GiB |
| LoRA of rank 16 on the attention projections | 11.8M | 85.2 ms | 114.3 ms | 208.2 ms | 26.48 GiB | 38.76 GiB |
| the first two layers | 201.9M | 62.5 ms | 84.2 ms | 147.3 ms | 20.36 GiB | 33.26 GiB |
| the last two layers | 201.9M | 61.7 ms | 10.2 ms | 72.4 ms | 5.31 GiB | 18.21 GiB |

The first row is the run of [Activations](#activations) with the optimizer added: 9 ms for the update and 15 GiB for the two moments.

Read the backward column.

- With only the last two layers trained, nothing below them needs a gradient. The backward pass takes 10 ms, and the 34 lower layers keep no activations.
- With only the first two layers trained, the gradient has to travel through all 36 layers to reach them. The backward pass takes 84 ms, 73% of the full one. The rule predicts half, dgrad without wgrad. The measured share is higher because parts of the backward pass have no weight gradient to skip: attention, the elementwise kernels and the loss.
- LoRA has a backward pass as long as the full model's. The gradient still passes through every frozen layer, which the row for the first two layers puts at 84 ms, and the adapters add a backward pass of their own. That the sum lands on the 116 ms of the full model is a coincidence of this shape.

The other columns have their own lessons. A frozen linear layer does not need its saved input, since dgrad uses only the weight, so the row for the first two layers holds 24% less after the forward pass than full fine-tuning does.

LoRA's forward pass is 22 ms longer, about 150 µs for each of the 144 adapted projections. The adapters are kept in fp32, so each one casts its input up, runs two small fp32 matmuls and casts the result back. That is far more time than the arithmetic of two small matmuls needs, which points at launches and memory traffic.

Its optimizer step takes 8.5 ms for 11.8M parameters, close to the 9 ms for all 4,022M, because the 288 small adapter tensors are updated one at a time. Both are the overhead-bound case of [Many small kernels](#many-small-kernels) again, and the step ends up 10% slower than full fine-tuning here.

LoRA saves state. Gradients and optimizer state are 0.13 GiB against 22.5 GiB, and the peak falls from 54.1 to 38.8 GiB. Whether the step also gets faster depends on the model and the shape. In Halo's benchmark of Qwen3-8B at 16,384-token rows with gradient checkpointing on, LoRA on the attention projections ran 18% faster than full fine-tuning at about half the memory. Its speed did not depend on the LoRA rank ([PEFT → Measured cost](../optimization/peft.md#measured-cost)).

![Stacked bars for one training step of Qwen3-4B on a B300 at 4,096 tokens, four cases in two columns. Step time split into forward, backward and optimizer: every parameter 63, 116 and 9 ms, 189 in all. LoRA adapters 85, 114 and 8.5, 208. The first two layers only 63 and 84, 147. The last two layers only 62 and 10, 72. Peak memory, split into what is held before the step, what the forward saves and what the backward adds: 54, 39, 33 and 18 GiB.](../assets/diagrams/finetune_cost.png)

Check yourself:

1. A model has 8B parameters, all trainable. How many GB do weights, gradients and bf16 moments take, before activations? And with fp32 moments?
2. Qwen3-4B has a hidden size of 2,560 and 36 layers, and a layer keeps about 33 hidden states' worth for the backward pass. How much do the decoder layers hold at 8,192 tokens without gradient checkpointing?
3. Gradient checkpointing makes a step of equal batch about 29% longer. Give two reasons why the largest batch that fits with it can still give more tokens/s than the largest batch that fits without it.

[Answers](#answers-on-memory) are at the end of the page.

## Many GPUs

One GPU has three limits on its speed and one on its capacity. Several GPUs add two to the first list: the time it takes to move data between them, and the time one of them waits for another.

### Processes, ranks and groups

A multi-GPU job is one process per GPU. A launcher such as `torchrun`, which you run once on each server, starts them. A few words describe where a process sits in the job.

- A **rank** is one process with its GPU. Ranks are numbered from 0 across the job.
- A **node** is one server. A rank's local rank is its number within its node.
- The **world size** is the number of ranks in the job.

The processes share nothing. Each has its own model object, its own optimizer, its own random-number stream and its own environment variables. Whatever has to agree between them is made to agree by a message, or by seeding them identically.

That has two consequences.

- Replicas of a model have to start identical. A framework either broadcasts one rank's weights to the others when training starts, or it relies on every rank being seeded alike. In the second case, a rank that draws anything from its own random stream stays different from its peers for the whole run, and no loss value shows it ([Data Parallelism → HSDP](../parallelism/data-parallelism.md#hsdp-hybrid-sharded-data-parallel)).
- Settings have to agree as well. An environment variable that changes how many transfers a rank takes part in, or how it cuts one up, and that is set on some nodes only, can hang the job or let the ranks drift apart silently. It does not show up as a slow run ([DeepEP → Environment variables](../infrastructure/deepep.md#environment-variables)).

A **process group** is a subset of the ranks with its own connections and buffers. Every coordinated transfer runs over one group. A job has several: all ranks, the ranks that are replicas of each other, the ranks that share one split tensor.

Take 16 ranks on 2 nodes of 8, numbered so that rank = node × 8 + local rank. Suppose each tensor is split 8 ways inside a node. The two groups that share a split tensor are ranks 0 to 7 and ranks 8 to 15. Ranks at the same position in their node hold the same piece, so they are replicas of each other: {0, 8}, {1, 9} and so on up to {7, 15}.

![Sixteen rank cells in two rows of eight, one row per node, each showing its rank and local rank. One outline surrounds all sixteen. A blue outline surrounds each row, the two split groups of ranks 0 to 7 and 8 to 15. Eight violet outlines surround the columns, the replica groups from 0 and 8 to 7 and 15.](../assets/diagrams/ranks_and_groups.png)

> In Halo: `halo/runtime/process_group.py` starts the group of all ranks. The smaller groups come from `halo/distributed/mesh.py` and from `build_groups` in `halo/runtime/collectives.py` ([Parallelism → Communication and data flow](../parallelism/README.md#communication-and-data-flow)).

### The step waits for the slowest rank

A coordinated transfer among the ranks of a group is called a **collective**, and the next section describes the kinds. A collective finishes when every rank of its group has entered it and done its part, so a rank that arrives early waits. The time of a step is therefore set by the slowest rank at each collective, not by the average rank.

Anything that makes one rank later than its peers costs every rank the difference. More tokens to process do it, and so do longer rows in its batch, a kernel that compiles, a stall on the rank's CPU or a worse placement of its CPU thread. This lecture calls that time **waiting**, and keeps it apart from communication, which is the time the bytes take to move.

Uneven work is the common cause. When the experts of an MoE layer are spread over 8 GPUs, each GPU processes the tokens routed to its experts, and routing is uneven. These factors are recorded ones, the first on gpt-oss-20b at batch 4 of 4,096-token rows:

| Input | Tokens on the busiest rank, against the mean |
|---|---|
| uniform random token ids | about 2.4 times |
| real text, gpt-oss-20b | 1.99 times |
| real text, Qwen3-30B-A3B | 1.40 to 1.72 times |

With the random ids the mean rank waits for 27 to 39% of the step, a share calculated from the timings of the step's phases. A run that forced even routing was 1.83 times faster, 24,515 against 13,399 tokens/s per GPU. Forcing the routing changes the model, so that run only shows how large the wait is ([Throughput Benchmarks → Where the EP step's time goes](../optimization/throughput-benchmarks.md#where-the-ep-steps-time-goes)).

Rows of uneven length have the same effect on your ranks. The `transformers` library has a sampler for that reason, `BatchRebalanceSampler`, which deals rows of about equal cost to every rank ([Distributed Data Loading](../parallelism/data-loading.md#how-the-dataloader-is-built)).

A waiting rank looks busy from outside. It reads 100% utilization at the power of an idle process, 238 W on a B300. It is the last row of the table in [Power and utilization](#power-and-utilization), and the fifth check of [A method for one step](#a-method-for-one-step) says how to find the wait on your own job.

### Collectives

Four patterns of collective cover training. Take a group of `R` ranks and a tensor of `D` bytes. `D` is the size of the whole tensor, which is the amount each rank ends with in an all-gather and the amount each rank starts with in the other three.

![Grids of four ranks by four pieces before and after each collective. All-gather: one piece each, then every piece on every rank. Reduce-scatter: full copies, then one summed piece each. All-reduce in three grids: full copies, sums on the diagonal after a reduce-scatter, sums everywhere after an all-gather, (R − 1)/R × D on each arrow. All-to-all: a transpose, each piece one hop to its one destination, (R − 1)/R × D per rank.](../assets/diagrams/collective_ops.png)

The figure also names where training uses each collective, with terms that [Data parallelism](#data-parallelism) explains. The table gives the same four in bytes:

| Collective | Before | After | Bytes each rank sends, and as many it receives |
|---|---|---|---|
| all-gather | each rank holds one piece, `D/R` | every rank holds all of `D` | `(R−1)/R × D` |
| reduce-scatter | each rank holds its own full `D` | each rank holds one piece of the sum | `(R−1)/R × D` |
| all-reduce | each rank holds its own full `D` | every rank holds the whole sum | `2 × (R−1)/R × D` |
| all-to-all | each rank holds `R` pieces, one per destination | each rank holds the pieces addressed to it | `(R−1)/R × D` |

The factor is `(R−1)/R` because each rank already holds one piece in `R`. These are the factors of the benchmark of **NCCL**, NVIDIA's collective library. They are the least that any algorithm built from transfers between pairs of ranks can move through a rank's link. A ring moves that much, and a tree moves more through its inner ranks.

Three relations hold between the four.

1. An all-reduce is a reduce-scatter followed by an all-gather.
2. The backward pass of a collective is its mirror. The gradient of an all-gather is a reduce-scatter and the reverse, and the gradient of an all-to-all is another all-to-all. A value that was copied to every rank receives a gradient from every rank, and those gradients have to be summed. So when you choose the collective of the forward pass, you have chosen the one of the backward pass.
3. An all-to-all and an all-gather move the same bytes per rank. They differ in the result and in the path. After an all-gather every rank holds the same `D`. After an all-to-all each rank again holds `D`, but a different one on each rank: `R` pieces of `D/R`, one from every rank, and each piece travelled one hop to its one destination.

#### What runs underneath

`torch.distributed` calls NCCL for every collective. On each GPU NCCL runs a small kernel that copies and reduces, with "one CUDA block per communication channel" and at most 64 channels. A collective is therefore itself a kernel of up to 64 blocks, and those blocks take places on the SMs that compute would use.

NCCL picks an algorithm by message size and by how the GPUs are connected. In the textbook model, a ring passes data around the group in `R − 1` rounds for each half of an all-reduce and uses the links well for large messages, and a tree needs about log₂ `R` rounds and suits small ones. NCCL's own ring and tree follow that pattern with different constants.

It also picks a protocol by size. Its environment variables name the choices: `NCCL_PROTO` takes `LL` and `LL128`, its two low-latency protocols, and `Simple`, the one for large messages. `NCCL_ALGO` takes Ring, Tree, NVLS and others.

You will meet **bus bandwidth** wherever collectives are measured. NCCL's benchmark defines it as the message size divided by the time, multiplied by the factor in the table. Its notes give the reason: "Using this bus bandwidth, we can compare it with the hardware peak bandwidth, independently of the number of ranks used." It is not the rate of one GPU's link. It is the link rate an ideal algorithm would have needed to finish in the measured time, so you can set it beside a link's specified rate.

#### Small messages and large ones

A collective costs a fixed time per call plus the bytes over the bandwidth. A small message is bound by the fixed time, and a large one by bandwidth. These all-reduces were measured through `torch.distributed` on 2 and 4 GPUs of one B300 node, connected by NVLink:

| Message size | 2 GPUs | 4 GPUs |
|---|---|---|
| 4 KiB | 19.3 µs | 27.2 µs |
| 1 MiB | 20.6 µs | 29.5 µs |
| 4 MiB | 36.6 µs, 115 GB/s bus bandwidth | 44.7 µs, 141 GB/s |
| 16 MiB | 53.1 µs, 316 GB/s | 96.8 µs, 260 GB/s |
| 64 MiB | 149 µs, 450 GB/s | 181 µs, 556 GB/s |
| 256 MiB | 525 µs, 511 GB/s | 673 µs, 599 GB/s |
| 1 GiB | 1,867 µs, 575 GB/s | 2,478 µs, 650 GB/s |
| 4 GiB | 7,020 µs, 612 GB/s | 9,543 µs, 675 GB/s |

Up to 1 MiB the time does not depend on the size. A collective costs 17 to 30 µs here whatever it carries, across all four kinds. That is the cost of one call through `torch.distributed`, CPU included, and this run cannot separate the latency of an NVLink hop from it. For the all-reduce the time starts to grow between 1 and 4 MiB. The other three kinds still take 29 to 33 µs at 4 MiB.

Bus bandwidth reaches half of its 4 GiB value at about 16 MiB on 2 GPUs and between 16 and 64 MiB on 4. At 4 GiB it is still rising. The other three collectives reach 444 to 490 GB/s on 2 GPUs and 608 to 622 GB/s on 4 at that size.

On all 8 GPUs of the node the shape is the same. An all-to-all costs 25 to 29 µs up to 1 MiB and reaches 651 GB/s at 4 GiB. An all-reduce with NVLink SHARP switched off, a feature that [The wires](#inside-the-nvlink-domain) describes, costs 33 to 34 µs up to 1 MiB and reaches 687 GB/s at 4 GiB (measured).

You can estimate where the two regimes meet as the fixed time multiplied by the bandwidth. With the 20 µs of a small call on 2 GPUs and 600 GB/s that is 12 MB (computed), close to the 16 MiB at which the measured bus bandwidth is half-way up. On 2 GPUs the factor of an all-reduce is 1, so its bus bandwidth is the plain size over time and the estimate applies as it stands.

![Two panels against message size from 4 KiB to 4 GiB for NCCL collectives on 2 and 4 GPUs of one B300 node, the all-reduce in colour and the other three as thin grey lines. Time per call on a log axis: 19 to 29 µs up to 1 MiB, then rising to 7,020 and 9,543 µs. Bus bandwidth on a linear axis: near zero up to 1 MiB, then rising to 612 and 675 GB/s, under a line at NVLink's specified 900.](../assets/diagrams/collective_by_size.png)

#### What to do about communication

While a GPU is inside a collective its tensor cores compute nothing. Three things reduce that time.

- Do more work per collective, which means more tokens per step.
- Keep each group on the fastest link, and no larger than it needs to be.
- Run the collective beside compute where its result is not needed at once.

A smaller group does not move many fewer bytes per rank. From 16 ranks to 8 the factor goes from 15/16 to 7/8, which is 6.7% less (computed). A smaller group helps in other ways: fewer rounds for small messages, a shorter wait for the slowest rank, and a better chance of staying on a fast link.

Check yourself:

1. An all-reduce of a 1 GiB tensor runs on 8 ranks. How many bytes does each rank send?
2. A collective of 64 KiB takes 20 µs on 2 GPUs. What would you expect for 256 KiB?
3. A group shrinks from 8 ranks to 4. By how much do the bytes per rank of an all-gather fall?

[Answers](#answers-on-collectives) are at the end of the page.

### The wires

Every link rate in this lecture is per GPU, per direction, in bytes per second. NVIDIA quotes NVLink and PCIe with the two directions added together, so those figures are twice the ones in this table. A network port's rate is already per direction.

The table uses one name you may not know. **EFA** is the network adapter of AWS's GPU nodes, and a 400-gigabit port carries 50 GB/s.

| Link | Connects | Rate per GPU, per direction | Kind of number |
|---|---|---|---|
| HBM | a GPU and its own memory | 6.6 TB/s for a copy, read plus write | measured |
| NVLink 5 | GPUs in one NVLink domain | 0.9 TB/s, from 18 links of 50 GB/s | specification |
| NVLink 5, a copy from one GPU to the next | 2 GPUs of one node | 767 to 779 GB/s at 1 and 8 GiB | measured |
| NVLink 5, as NCCL uses it | 2 and 4 GPUs of one node | 612 and 675 GB/s bus bandwidth for a 4 GiB all-reduce | measured |
| PCIe Gen6 ×16 | a GPU and the CPU on a B300 node | 128 GB/s raw | specification |
| the same link, as a copy from pinned CPU memory uses it | a GPU and the CPU | 56 to 57 GB/s | measured |
| EFA on a B300 node | GPUs on different nodes | 100 GB/s, from two 400-gigabit ports per GPU | specification |
| EFA, as NCCL uses it | one GPU pair across two nodes | 92.8 GB/s at 256 MiB and 96.2 GB/s at 1 GiB | recorded |
| one 400-gigabit InfiniBand port | GPUs on different nodes | 50 GB/s | specification |
| TCP sockets | the fallback across nodes | 9.7 GB/s for a weight broadcast | recorded |

Four rows carry more than the table shows:

- The copy over NVLink reaches 87% of the specified rate.
- The PCIe figure is the GPU's own link, and a copy from pinned CPU memory reaches under half of it.
- The EFA rates are for a send and a receive between one pair. Eight pairs at once reached 776 GB/s in all ([Pipeline Parallelism → Boundary cost](../parallelism/pipeline-parallelism.md#boundary-cost)).
- The socket rate is one stream between two ranks ([Rollout Servers](../infrastructure/rollout-servers.md#servers-on-other-nodes-efa)).

![Link rates per GPU and per direction on a log axis, dots for measured values and ticks for specified ones. Inside a node: HBM 6.65 TB/s measured and 8 specified, NVLink 5 767 to 779 GB/s measured and 900 specified, PCIe Gen6 ×16 56 to 57 measured and 128 specified. Across nodes: EFA 93 to 96 measured and 100 specified, one 400-gigabit InfiniBand port 50 specified, TCP sockets 9.7 measured. Beside each, the time to move 16 GB, from 4.8 ms to 1.6 s.](../assets/diagrams/interconnect_tiers.png)

On this convention NVLink is 9 times the network of these nodes and 18 times one 400-gigabit port. NVIDIA's own table for an 8-GPU system gives the same 9: 14.4 TB/s of NVLink against 1.6 TB/s of networking, both counted two-way.

In time, the 16 GB of bf16 weights of an 8B model cross NVLink in 18 ms and the network in 160 ms (computed). On TCP sockets, one gather of the weights at the scale of gpt-oss-20b took about 15 s ([Data Parallelism](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward)). That gather is a collective among all the ranks, and it runs well under the 9.7 GB/s that one stream between two ranks reaches.

#### Inside the NVLink domain

NVLink connects GPUs to each other without the CPU or PCIe in the path. On an 8-GPU board the path goes through a switch chip, the NVSwitch, and every GPU reaches every other at the full link rate.

The **NVLink domain** is the set of GPUs that reach each other over NVLink. On an 8-GPU server it is the node. On a GB200 or GB300 NVL72 rack it is 72 GPUs in 18 compute trays, so the boundary is the rack and not the node.

The switch can also do arithmetic. With **NVLink SHARP**, NVLS for short, the switch computes the sum of an all-reduce, so each GPU sends its tensor once and receives the result once. Against the `2 × (R−1)/R` of the table that is at best 1.75 times less traffic at 8 GPUs, approaching 2 with more (computed).

NCCL supports it from version 2.17.1 and turns it on where the switch supports it. One figure from elsewhere, in a footnote of Aleksa Gordić's article in the reading list, is about 1.3 times on a 4 GB all-reduce on 32 H100s.

#### Across nodes

With **RDMA**, remote direct memory access, the network card reads and writes a machine's memory without the operating system in the path. GPUDirect RDMA extends that to GPU memory, so a transfer needs no copy through the CPU's memory. Without a working RDMA network, NCCL falls back to TCP sockets through the CPU's memory, and it does so without an error.

The fallback is expensive. Qwen3-8B on 2 nodes of 2 GPUs ran 14,794 tokens/s per GPU over EFA and 2,024 on sockets, 7.3 times apart ([Data Parallelism](../parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync)). If a multi-node run is several times slower than you expect, check which transport NCCL chose. Its startup log says so when you set `NCCL_DEBUG=INFO`.

Bandwidth is not the only property of a link. Across two of these nodes an NCCL all-to-all reaches about 173 GB/s bus bandwidth, part of which never leaves a node. The transfer that carries tokens to experts on other GPUs, measured at every step on the same nodes, takes about 100 times what that bandwidth predicts, because it consists of many small operations and each has its own latency ([DeepEP → Expert parallelism over AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa)).

So keep a collective that runs at every layer, and whose result the next operation needs, inside the NVLink domain. Whatever crosses nodes should happen once per step, or go point to point.

Nodes also do not share a filesystem, an environment or caches unless you arrange it ([Multi-Node → Shared vs node-local storage](../parallelism/multi-node.md#shared-vs-node-local-storage)).

> In Halo: the size of the NVLink domain is declared and checked against the hardware ([Multi-Node](../parallelism/multi-node.md)).

### Data parallelism

**Data parallelism** gives every rank the same model and a different batch. Each such rank is a **replica**. It is the starting point, and each of the other modes in this part is combined with it.

Each rank computes the mean loss of its own batch, and the gradients of the `R` ranks are averaged. That average equals the gradient of the mean loss over all tokens only if every rank has the same number of loss tokens, the tokens that count in the loss. If yours do not, a token on a rank with few loss tokens counts for more than a token on a rank with many.

#### DDP and FSDP

**DDP**, PyTorch's `DistributedDataParallel`, is the plain form. Every rank holds the whole model, the whole gradient and the whole optimizer state, so no rank holds less than it would alone. The gradient is all-reduced once per backward pass. With `P` parameters at 2 bytes each, that is `2 × (R−1)/R × 2P` bytes per rank.

DDP sends it in buckets of about 25 MiB, each launched as the backward pass fills it, so most of the transfer runs beside the rest of the backward pass. DDP also broadcasts the first rank's weights to the other ranks when training starts.

**FSDP**, fully sharded data parallel, stores the model in pieces called shards. FSDP2 is PyTorch's current implementation of it. It handles a few consecutive layers at a time, a **layer group**, and for each one it runs this cycle:

1. At rest, each rank stores `1/R` of the group's weights.
2. Before the group's forward pass, an all-gather gives every rank a whole temporary copy.
3. The forward pass runs on the whole copy.
4. The backward pass produces a whole gradient on every rank.
5. A reduce-scatter leaves each rank with its `1/R` shard of the averaged gradient, and the whole copies are freed.
6. The optimizer updates the shards.

The cycle is DDP's all-reduce pulled apart around the compute. The reduce-scatter half runs after the backward pass, and the all-gather half before the next forward pass.

Unlike DDP, FSDP2 does not broadcast one rank's weights at the start. Plain sharding does not need it, because every rank gathers the same shards. Where whole copies exist, across the replicas of hybrid sharding or for a parameter kept outside the shards, the copies rely on identical seeding unless the framework copies one of them to the rest.

What happens between steps 3 and 4 is a choice, and its two options are named after the levels of DeepSpeed's ZeRO, the Zero Redundancy Optimizer. ZeRO-1 shards the optimizer state, ZeRO-2 the gradients as well, and ZeRO-3 the parameters as well.

In FSDP, keeping the gathered copy from a layer group's forward pass to its backward pass is the analog of ZeRO-2. Freeing it after the forward pass and gathering it again for the backward pass is the analog of ZeRO-3. That runs three collectives where there were two, 1.5 times the bytes. PyTorch's `fully_shard` defaults to the ZeRO-3 analog for every layer group but the root module.

#### Who holds what

The table counts bytes per rank, with bf16 weights and bf16 moments (computed from the cycle).

| | DDP | FSDP, ZeRO-2 analog | FSDP, ZeRO-3 analog |
|---|---|---|---|
| weights at rest | `2P` | `2P/R` | `2P/R` |
| gathered copy at the peak of the step | none | `2P`, the whole model | one or two layer groups |
| gradient while a layer group's backward pass runs | `2P` in all | whole for that group | whole for that group |
| gradient after the reduction | `2P` | `2P/R` | `2P/R` |
| optimizer state | `4P` | `4P/R` | `4P/R` |
| bytes on the wire per backward pass | `2 × (R−1)/R × 2P` | the same | `3 × (R−1)/R × 2P` |

![Eight small grids, one per mode, each with four rank columns and rows for weights, gradients, optimizer state, activations and tokens processed. A cell is filled whole, filled as one piece of four, split into a whole band over a piece, or empty. Plain data parallel is all whole, and FSDP at rest holds a piece of the weights, gradients and optimizer state and no activations. Under each grid are the collectives the mode adds.](../assets/diagrams/who_holds_what.png)

A split cell in the figure means that a rank holds part of the object whole and the rest as a piece. Under the ZeRO-2 analog that is the gradient: whole for the layer group whose backward pass is running, and a shard for the layer groups already reduced. Read the grids for data parallelism now. The others belong to the modes that follow, and [Combining the modes](#combining-the-modes) comes back to the figure.

**Under the ZeRO-2 analog every rank holds a whole copy of the weights at the peak of the step.** Adding ranks shrinks the optimizer state and the reduced gradients. It does not shrink that copy, and it does not shrink activations, because each rank runs its own batch. Under that setting the weights of your dense model have to fit on one GPU, however many GPUs the job has.

The ZeRO-3 analog lifts that limit for 1.5 times the traffic. A rank then holds its shards and one or two gathered layer groups.

Take Qwen3-8B as a worked number. It has 8.19B parameters, 16.4 GB in bf16. Per backward pass each rank moves 2 × 7/8 × 16.4 = 28.7 GB on 8 ranks and 30.7 GB on 16 (computed). FSDP moves the same bytes as DDP when weights and gradients travel in the same format. If the gradient is reduced in fp32, its half doubles.

#### What it costs in time

Part of FSDP's communication runs beside compute. The reduce-scatter of one layer group runs on its own stream while the next one's backward pass computes.

The all-gather of the forward pass also goes onto a stream of its own, when the CPU reaches the layer group. The CPU runs ahead of the GPU, so on a step that is not overhead-bound the all-gather runs beside the previous layer group's forward pass, which PyTorch calls implicit prefetching. On an overhead-bound step the CPU arrives too late and the forward pass waits. A setting, `set_modules_to_forward_prefetch`, issues the all-gather earlier. Under the ZeRO-3 analog the backward pass gathers again, and FSDP2 starts the next layer group's all-gather early there by default.

When the model fits on one GPU, plain data parallelism is the fastest layout. On Qwen3-8B it leads every other layout measured, on one 8-GPU B300 node and on two ([Pipeline Parallelism → Measured throughput](../parallelism/pipeline-parallelism.md#measured-throughput)).

**Hybrid sharding**, HSDP, is the variant for several NVLink domains. It shards inside each domain and replicates across domains. After the reduce-scatter inside its domain, each rank all-reduces only the shard it kept, `2P/R` bytes for `R` ranks per domain, with its counterparts in the other domains. For Qwen3-8B on two domains of 8 that is about 2 GB per rank across the slower link, where one 16-rank group moves 30.7 GB (computed). In exchange, each domain keeps its own copy of the parameters.

#### Gradient accumulation

Gradient accumulation runs several forward and backward passes per optimizer step and adds their gradients. It gives you a larger batch per optimizer step than fits in memory at once. Each of those passes is a **microbatch**. By default FSDP2 reduces the gradient after every microbatch and gathers the weights again, so accumulation multiplies the communication. Reducing once per optimizer step removes the reduce-scatters of all but the last microbatch. The matching all-gathers stay until the weights are also kept gathered across the step.

These gains from reducing once per step were recorded on B300s, on packed 4,096-token rows at batch 2 with 4 microbatches and gradient checkpointing on. The two one-node rows are single runs and the two-node row is the mean of three ([Data Parallelism](../parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync)):

| Run | Gain | Peak memory |
|---|---|---|
| Qwen3-8B, one 8-GPU node | +5.0% | 32.6 to 45.9 GiB |
| Qwen3-8B, 2 nodes of 2 GPUs over EFA | +8.9% | 40.2 to 51.7 GiB |
| Qwen3-30B-A3B with its experts not split, one 8-GPU node | +12.4% | 95.3 to 145.0 GiB |

The gain is larger where the link is slower: +5.0% on NVLink, +8.9% over EFA, and +52% on TCP sockets in one run. With hybrid sharding the two-node run gained 13.1%, also in one run. The memory column is the cost. One unsharded gradient is held from the first microbatch of the step to the last. With one microbatch per step there is nothing to save.

> In Halo: the default is the ZeRO-2 analog. A layer group, which Halo calls a shard group, is two decoder layers, and one under the ZeRO-3 analog ([Data Parallelism → Shard groups](../parallelism/data-parallelism.md#shard-groups)). The ZeRO-3 analog and the once-per-step reduction are settings, and some combinations of modes reject them ([Data Parallelism → ZeRO-2 vs ZeRO-3](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward)).

Check yourself:

1. An 8B model trains under FSDP on 8 ranks. How many bytes does each rank move per backward pass?
2. The same job switches to the ZeRO-3 analog. How many bytes does each rank move now, and what does the job get in return?
3. A job goes from 8 to 16 ranks under the ZeRO-2 analog. Which of these shrink on each rank: optimizer state, the gathered copy of the weights, activations?

[Answers](#answers-on-data-parallelism) are at the end of the page.

### Tensor parallelism

**Tensor parallelism** splits the weight matrices themselves across the ranks of a group.

Take two linear layers in a row with a function between them, `Y = X @ W₁` and then `Z = f(Y) @ W₂`. Split `W₁` by columns and `W₂` by rows. Each rank holds one block of columns of `W₁` and the matching block of rows of `W₂`. It computes its slice of `Y` from the whole input with no communication, applies `f` to that slice and multiplies by its rows of `W₂`. That works because `f` acts on each value on its own, as an activation function does. What it gets is a partial `Z` of full width. The true `Z` is the sum of the partials over the group, and that sum is one all-reduce.

In a transformer the column split falls on the query, key and value projections of attention and the row split on its output projection. In the MLP the two fall on the up projection and the down projection.

![Two rows, one per rank. In each, the whole input X is multiplied by that rank's half of W₁, cut by columns, giving its half-width slice of Y. f of the slice is multiplied by the rank's half of W₂, cut by rows, giving a partial Z of full width. One all-reduce joins the two rows and returns the summed Z to both ranks.](../assets/diagrams/tensor_parallel_split.png)

The ranks of a group share one batch, because every one of them has to see the same input. Tensor parallelism therefore divides the number of replicas by the size of the group.

**If you give the ranks of a group different batches, nothing raises an error.** When the batches differ in shape, the job stops at the first all-reduce and stays there until a watchdog ends it ([Tensor Parallelism → Identical-input requirement](../parallelism/tensor-parallelism.md#identical-input-requirement)). When they have the same shape, the all-reduce completes and sums partial results of different inputs, so the run is wrong with no error.

The mode adds two all-reduces per layer in the forward pass, one after attention and one after the MLP, and two in the backward pass. The Megatron authors, whose papers are the references for tensor and pipeline parallelism, describe them as tensors of batch × sequence × hidden size "all-reduced among t model replicas twice each in the forward and backward pass".

The forward all-reduce sits at the output of the row-split layer. The backward one sits at the input of the column-split layer: each rank computes a partial `dX` through its own columns of `W₁`, and the true `dX` is their sum. The gradient of the forward all-reduce itself needs no transfer, because every rank already holds the same `dZ`.

Each all-reduce carries a `[T, H]` activation. At 8,192 tokens and a hidden size of 4,096 in bf16 that is 67.1 MB. In a group of 2 the all-reduce factor `2 × (R−1)/R` is 1, so four all-reduces over 36 layers come to 4 × 36 × 67.1 MB = 9.7 GB per rank per step (computed). The total is a third of the 28.7 GB that FSDP moves for the same model. But none of it can run beside compute, because the next operation needs the sum.

In a group of `R` ranks, a rank holds `1/R` of the split weights, with their gradients and optimizer state. The projections of attention and of a dense MLP are always split. Whether the embedding and the output matrix are split too depends on the implementation, and Megatron splits them along the vocabulary. Activations are split in part as well: a rank holds its slice of `Y` and a partial `Z` of full width.

It has to stay inside the NVLink domain, since its all-reduces sit on the forward path of every layer. The Megatron authors put it as "tensor model parallelism is best within a node".

In the one recorded comparison the split ran at 0.59 times the rate of plain replicas. On Qwen3-8B across two 8-GPU B300 nodes, 8 replicas of a 2-way split ran 10,780 tokens/s per GPU where 16 plain replicas ran 18,337 ([Pipeline Parallelism → Measured throughput](../parallelism/pipeline-parallelism.md#measured-throughput)). Two GPUs under a 2-way split share one batch, so together they deliver 1.18 times the tokens of one GPU. How much of that gap is the all-reduce itself has not been measured.

Use tensor parallelism to fit a model that does not fit otherwise. It will not make a model that fits go faster.

The same split can also be applied inside the experts of an MoE layer, and [Expert parallelism](#expert-parallelism) ends with that variant.

From here on a layout is written as a product of modes in code font. `dp16` is 16 replicas, and `dp8 × tp2` is 8 replicas of a 2-way tensor split. In the sections that follow, `ep8` is a group of 8 ranks that share the experts, `cp2` a 2-way split of the context, and `pp2` a pipeline of 2 stages.

> In Halo: tensor parallelism is applied to unmodified `transformers` models ([Tensor Parallelism](../parallelism/tensor-parallelism.md), [Why this framework](why-this-framework.md)). The plan for dense models also splits the output matrix along the vocabulary. The plan for MoE models splits attention only and leaves the embeddings, the output matrix and the experts whole on every rank. The all-reduce is not overlapped with compute.

### Expert parallelism

**Expert parallelism** spreads the experts of each MoE layer over the ranks of a group. With `R` ranks, each rank holds `E/R` whole experts with their gradients and their optimizer state. Every rank runs its own batch, so this mode does not reduce the number of replicas.

For each MoE layer the work has four steps.

1. On each rank the router picks `k` experts for each of the rank's own tokens.
2. A **dispatch** sends each token to the ranks that hold its chosen experts.
3. Each rank runs its experts on the tokens it received.
4. A **combine** sends the results back to the rank each token came from, where they are added up with the router's weights.

Dispatch and combine are all-to-alls, and this lecture calls the pair the **token exchange**. The backward pass runs the pair in reverse, so one MoE layer costs four all-to-alls per microbatch. A token can go to as many as `k` ranks, and the pieces of the all-to-all are uneven because routing is uneven.

![Expert parallelism for one MoE layer. Each rank routes the tokens of its own batch to their top-k experts. A dispatch all-to-all sends every token to the rank that holds its expert. Each rank runs its local experts on what it received as one grouped GEMM. A combine all-to-all returns each result to the rank the token came from.](../assets/diagrams/ep_token_routing.png)

The figure's labels write EP for expert parallelism and name DeepEP, a library that implements the token exchange.

The tokens of the whole group pool at the experts, which changes the row count of an expert's GEMM:

```text
M ≈ R × tokens per rank × k / E
```

`M` grows with the group. For gpt-oss-20b over 8 ranks, at batch 4 of 4,096-token rows, it is 8 × 16,384 × 4 / 32 = 16,384 rows per expert (computed). A smaller group gives each of your experts fewer rows at the same batch per GPU.

In memory a rank holds `1/R` of the expert weights, with their gradients and optimizer state. Weights outside the experts are not touched by this mode, and neither are a rank's activations, since each rank runs its own batch. The mode adds a receive buffer that grows with the group. In DeepEP the buffer is about `R` × tokens per rank × (2 × `H` + 4 × `k`) bytes. Over 8 ranks at 8,192 tokens per rank, a hidden size of 4,096 and 10 experts per token that is 0.5 GiB ([DeepEP → Buffer sizing](../infrastructure/deepep.md#buffer-sizing)).

#### Where the time goes

The table gives each of the three phases of an MoE layer, dispatch, expert compute and combine, as a share of their sum. The rest of the step is not in it. The shares cover the forward pass and its recomputation under gradient checkpointing, as a mean over the ranks. They were recorded on 8 B300s with uniform random token ids ([Throughput Benchmarks → Where the EP step's time goes](../optimization/throughput-benchmarks.md#where-the-ep-steps-time-goes)):

| Group | Dispatch | Expert compute | Combine, with its wait |
|---|---|---|---|
| 8 ranks, gpt-oss-20b | 9 to 13% | 46 to 47% | 39 to 45% |
| 2 ranks, gpt-oss-20b and Qwen3.5-35B-A3B | 12% | 72 to 74% | 13 to 16% |

![Stacked bars of the shares of the three phases of an MoE layer: dispatch, expert compute and combine. With experts over 8 ranks, combine with its wait takes 39 to 45% and expert compute 46 to 47%. With experts over 2 ranks, combine takes 13 to 16% and expert compute 72 to 74%.](../assets/benchmarks/ep_phase_split.png)

With 8 ranks, most of a rank's combine is its wait for the rank with the most rows, the effect described in [The step waits for the slowest rank](#the-step-waits-for-the-slowest-rank). With 2 ranks expert compute dominates. The dispatch is the smallest phase in both.

Balancing the router is aimed at that wait. One method is an auxiliary loss. Another is DeepSeek-V3's bias per expert, which is lowered for overloaded experts and raised for underloaded ones. How far either lowers the wait on real text has not been measured.

#### Which link it runs on

You may let the exchange cross nodes, and it is slow there. On two 8-GPU B300 nodes over EFA, one group of 16 ran 4,115 tokens/s per GPU. A group of 8 inside each node, with the two nodes as replicas, ran 18,685. The ratio is 0.22 ([Pipeline Parallelism → Measured throughput](../parallelism/pipeline-parallelism.md#measured-throughput)). The exchange consists of many small operations, so across nodes it is bound by the latency of each operation, as [The wires](#across-nodes) showed.

Inside one node the split has a cost of its own at small batch. At batch 1 on gpt-oss-20b, a group of 8 ran 17% below the same model with no split, 11,896 against 14,304 tokens/s per GPU, for 2.3 times less memory ([Throughput Benchmarks](../optimization/throughput-benchmarks.md#where-the-ep-steps-time-goes)).

**Expert-tensor parallelism** applies the column and row split of [Tensor parallelism](#tensor-parallelism) inside each expert MLP. The tokens travel to the ranks that hold their experts as before. With the split added, an all-reduce runs on the tokens after they have come back.

> In Halo: the exchange runs on DeepEP, DeepSeek's library for this all-to-all ([Expert Parallelism](../parallelism/expert-parallelism.md), [DeepEP](../infrastructure/deepep.md)). The balancing methods are settings ([Callbacks → MoE balancing modes](../training-methods/callbacks.md#moe-balancing-modes)). Expert-tensor parallelism has its own page ([Expert Tensor Parallelism](../parallelism/expert-tensor-parallelism.md)).

### Context parallelism

**Context parallelism** splits one long row across the ranks of a group. With `R` ranks, each rank holds `S/R` consecutive tokens and all attention heads. Everything outside attention works on tokens one at a time, so it runs on the local piece with no communication.

Attention needs every token of the row. Around each attention layer, one all-to-all turns "my tokens, all heads" into "all tokens, my share of the heads", for the queries, keys and values together. Attention runs. A second all-to-all turns the output back. The exchange is exact, because the heads of an attention layer are independent of each other. It deals whole heads to the ranks, so the size of the group has to divide the number of query heads and the number of key and value heads. The scheme is called Ulysses, after the DeepSpeed-Ulysses paper.

![Ulysses attention on four ranks. Position encoding runs on each rank's local piece of the row. One all-to-all on the queries, keys and values gives every rank the whole row for a quarter of the heads. Attention runs on that. A second all-to-all on the output gives every rank back its own piece of the row with all heads.](../assets/diagrams/ulysses_attention_flow.png)

The figure writes CP for context parallelism, RoPE for the rotary position encoding and `o_proj` for the output projection of attention.

The mode adds two all-to-alls per attention layer in the forward pass and two in the backward pass. Sending queries, keys and values in one exchange and not in three gained 4.4% on Qwen3-8B with a 2-way split at 4,096-token rows on 8 B300s, recorded on Halo 1.2.0 ([Context Parallelism](../parallelism/context-parallelism.md#how-ulysses-works)). A whole-step rate for this mode against plain replicas is not on record.

The ranks of a group share one batch, so context parallelism divides the number of replicas by the size of the group. Each rank computes a full-size gradient from its own piece of the row, and the gradient of the whole row is the sum over the group. The gradients therefore have to be reduced across the group even when the job has a single replica.

The loss needs two corrections. The target of the last token of each piece is the first token of the next piece, which is on another rank. And the loss needs one normalizer for the whole row, because the pieces hold different numbers of loss tokens.

Under context parallelism a rank holds `1/R` of the activations of the row. The weights, their gradients and the optimizer state stay whole. It is the one mode that divides the row itself, so it is the one to reach for when a single row of yours does not fit. Pipeline parallelism also lowers the activations a rank holds, by a different route that the next section describes.

It belongs inside the NVLink domain, because its all-to-all carries full activations at every layer.

Megatron has a different technique with a similar name. Its sequence parallelism takes the parts of a layer that tensor parallelism leaves whole, the norms and dropout, and splits them along the sequence across the ranks of the tensor-parallel group.

> In Halo: each rank's loss is `R × local sum / global tokens`, and the gradient average over the group cancels the `R`. Context parallelism cannot be combined with tensor parallelism ([Context Parallelism](../parallelism/context-parallelism.md)).

Check yourself:

1. Tensor parallelism in a group of 4 runs two all-reduces of a 16,384 × 4,096 bf16 activation per layer in the forward pass and two in the backward pass, over 36 layers. How many bytes does each rank move per step?
2. A model has 128 experts and routes each token to 8. The experts are spread over 4 ranks with 8,192 tokens per rank. How many rows does each expert's GEMM have?
3. A row of 65,536 tokens is split 4 ways with context parallelism. How many tokens does a rank hold, and what happens to its activations and to its weights?

[Answers](#answers-on-context-parallelism) are at the end of the page.

### Pipeline parallelism

**Pipeline parallelism** splits the stack of layers into `p` consecutive **stages**, with one group of ranks per stage. A stage here is a run of layers, and it has nothing to do with the buffer stages inside a GEMM kernel. The batch of a step is cut into `m` microbatches that flow through the stages one after another. Every other mode runs unchanged inside one stage, on world size / `p` ranks.

The lowercase `p` and `m` are counts. They are different letters from `P`, the parameters of the model, and `M`, the rows of a GEMM.

#### The boundary

The mode adds one send and one receive between two ranks, per boundary and per microbatch. A stage boundary carries one tensor per microbatch, the hidden state on the way forward and its gradient on the way back. Over a step, in bf16, one boundary carries:

```text
bytes across one boundary per step = 4 × T × H      (2 bytes, 2 directions)
```

At 8,192 tokens and a hidden size of 4,096 that is 134 MB, and at 32,768 tokens it is 537 MB. FSDP moves 28.7 to 30.7 GB per backward pass for the same model. The boundary is 210 to 230 times less at the first size and 53 to 57 times less at the second (computed).

Compare that with the other modes. Tensor, context and expert parallelism send a tensor of about that size at every layer, and FSDP sends every parameter. A pipeline sends one tensor per boundary. So a pipeline keeps most of its speed on a link that is too slow for the others.

Two measurements on Qwen3-8B with two 8-GPU B300 nodes show both sides, at 4,096-token rows with 8 microbatches and gradient checkpointing off. Over EFA, 16 plain replicas ran 18,337 tokens/s per GPU and two pipeline stages of 8 ran 17,608, 4.0% less. With the same nodes forced onto TCP sockets, the pipeline kept 0.875 of its rate and the replicas kept 0.061, which put the pipeline 13.8 times ahead ([measured rows](../parallelism/pipeline-parallelism.md#measured-throughput)).

A stage still runs FSDP inside itself, on half the parameters and on NVLink. The pipeline confines that traffic to the fast link. It does not remove it.

#### The bubble

While the pipe fills at the start of a step and drains at the end, stages wait. Each rank is idle for `p − 1` of the step's `m + p − 1` microbatch slots. The idle share is called the **bubble**:

```text
bubble = (p − 1) / (m + p − 1)
```

For `p = 4` and `m = 8` it is 3/11, or 27%.

The idle slots fall in four places. Later stages wait before their first forward pass arrives. Every stage but the last waits between its first few forward passes and the first backward pass coming back. Every stage but the last waits again between the last backward passes, as the pipe drains. And the later stages finish their last backward pass early and wait for the earlier ones.

Two schedules are in use. **GPipe** runs all forward passes and then all backward passes. **1F1B** alternates one forward and one backward once the pipe is full.

![Two grids of 4 stages by 22 slots for p = 4 and m = 8. Under 1F1B brackets mark the fill over slots 0 to 6, the steady state with every stage busy over slots 7 to 15 and the drain over slots 16 to 21. Under GPipe every stage runs all forwards, then all backwards. Each stage idles 6 of 22 slots in both. A stage holds 4, 3, 2 and 1 live microbatches under 1F1B and 8 under GPipe.](../assets/diagrams/pp_bubble.png)

The figure counts a forward and a backward slot for each microbatch, so for `p = 4` and `m = 8` each stage is idle for 6 of 22 slots, the same 27%.

The bubble depends only on the ratio of `m` to `p − 1`, since bubble = 1 / (1 + `m` / (`p − 1`)). The table computes it for a few cases.

| | `m` = 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| `p` = 2 | 33% | 20% | 11% | 5.9% | 3.0% | 1.5% |
| `p` = 4 | 60% | 43% | 27% | 16% | 8.6% | 4.5% |
| `p` = 8 | 78% | 64% | 47% | 30% | 18% | 9.9% |

Read it along a row. Doubling `m` cuts a large bubble by about a fifth and a small one by about half. With `m = 2 × p` the bubble runs from 1/5 at `p = 2` toward 1/3 for many stages. With `m = 9 × (p − 1)` it is 10%. GPipe's rule of thumb is that the bubble is negligible when `m ≥ 4 × p`.

More stages at a fixed step size make the bubble worse, because the step has only so many rows to cut into microbatches. With 8 microbatches, 8 stages have a 47% bubble. Interleaved schedules, described in the Megatron pipeline paper, shrink it.

You will meet a second definition of the bubble. Megatron divides the idle time by the ideal time, which gives `(p − 1) / m`: 37.5% for the case that is 27% here.

The formula assumes that every slot costs the same. A backward pass costs twice a forward pass, `4 × P` against `2 × P`, and the idle share does not change: for `p = 4` and `m = 8` it is 9 of 33 time units, the same 27% (computed).

#### Microbatches and M

Under a pipeline the microbatches take the place of gradient accumulation. More stages need more microbatches to keep the idle share down, and each added microbatch makes every GEMM smaller. Choose `m` with both in view.

PyTorch's pipeline schedule sizes the buffers between stages on the first step, so every batch has to have one row length. Padding every batch to the maximum length gives you that, and packing keeps the padded rows full.

#### What a stage holds

A stage holds `P/p` parameters. PyTorch's pipeline schedule keeps them gathered across the backward pass and defers the gradient reduction to the end of the step. Each rank therefore holds 2 bytes of weight and 2 bytes of accumulating gradient for every parameter of its stage, or 4 bytes of gradient if the reduction runs in fp32. Its optimizer state stays sharded, at 4 bytes × `P` / world size.

At `p = 2` the weights plus the held gradient equal the whole gathered copy of FSDP, so two stages save nothing on the first stage. The first stage peaks at 0.98 of the memory of the run without a pipeline on Qwen3-8B and 0.84 on gpt-oss-20b, and the last stage at 0.68 and 0.45 (recorded with gradient checkpointing off, [Pipeline Parallelism → Measured throughput](../parallelism/pipeline-parallelism.md#measured-throughput)). The run without a pipeline is plain data parallelism for Qwen3-8B, and a group of 8 per node sharing the experts for gpt-oss-20b.

A pipeline saves activation memory. A stage processes every token of the batch, but it holds only a few microbatches at a time. Write `L` for the number of layers and `B` for the rows in a rank's batch, and count the stages from 0. Under 1F1B, stage `i` has at most `p − i` microbatches in flight, over `L/p` layers, at `B/m` rows each. For the first stage that is `p` microbatches over `L/p` layers at `B/m` rows, `L × B / m` in all.

It is `1/m` of a run that pushes all `B` rows through one forward pass, and the same as a run that accumulates gradients over the same `m` microbatches. GPipe holds all `m` microbatches on every stage, which is the reason 1F1B is the usual schedule.

Gradient checkpointing still applies. Under a 2-stage pipeline, turning it off raised the rate by 18.9% on Qwen3-8B and 22.8% on gpt-oss-20b, for 1.9 and 2.1 times the peak memory. Two larger models, Qwen3.8-Flash-Next and Nemotron 3.5 Super VL, ran out of memory with it off.

#### When a pipeline is worth it

A pipeline is for a slow link, for a model whose alternative would send attention or experts across nodes, and for a model that does not fit on one node. For a model that fits on one node, on a fast link, it does not help: the two-node pipeline ran 4.0% below plain replicas.

On one 8-GPU node a dense pipeline ran 8.5% below plain replicas. A pipeline with experts split inside each stage ran 9.2% above the expert split alone on that node, because halving the group of the token exchange saved more than the bubble cost. The one-node runs had 8,192-token rows and 16 microbatches, one run each, and their page cautions that both gpt-oss-20b rates were still rising when they were read.

A pipeline schedule runs many backward passes per optimizer step, and that breaks anything written for one. A gradient hook that reduces after each backward pass is an example, and so is a loss normalizer that assumes a single one.

> In Halo:
>
> - A stage must cover a whole number of NVLink domains. With one domain per stage, every collective inside the stage stays on NVLink and only the boundary tensor crosses the slower link ([Pipeline Parallelism → Boundary cost](../parallelism/pipeline-parallelism.md#boundary-cost)).
> - The two shipped recipes have `p = 2` and `m = 8`, a bubble of 1/9, and Halo warns when `m < 4 × p`.
> - 1F1B is the default schedule, and no interleaved schedule is implemented.
> - The combinations it rejects and the collator a pipeline needs are on [Pipeline Parallelism](../parallelism/pipeline-parallelism.md#rejected-combinations).

Check yourself:

1. A pipeline has 4 stages and 8 microbatches. How large is the bubble? How many microbatches give a 10% bubble?
2. You raise `m` from 8 to 32 on those 4 stages at a fixed batch. What happens to the bubble, and what happens to the `M` of every GEMM?
3. How much crosses one stage boundary per step at 16,384 tokens and a hidden size of 2,560?

[Answers](#answers-on-pipeline-parallelism) are at the end of the page.

### Combining the modes

The modes combine, and the first thing you have to get right when they do is the number of replicas. Ranks that share a batch are not replicas of each other. Tensor parallelism, context parallelism, expert-tensor parallelism and the stages of a pipeline all share a batch. Expert parallelism does not. The number of replicas is the world size divided by the number of ranks that share one batch.

Each mode frees one kind of memory and adds one kind of collective to every step. The first table is the memory side, and the figure in [Who holds what](#who-holds-what) draws it as grids.

| Mode | Splits | Shares a batch | Frees |
|---|---|---|---|
| DDP | nothing | no | nothing |
| FSDP | parameters at rest, gradients, optimizer state | no | optimizer state and reduced gradients |
| hybrid sharding | the same, inside each domain | no | the same, and each domain keeps its own copy of the parameters |
| tensor | attention and MLP weights, and in some implementations the embedding and the output matrix | yes | those weights |
| expert | experts | no | expert weights |
| context | the row | yes | activations |
| pipeline | the layers | yes, along a chain | parameters and activations of the other stages |

The second table is the collective each mode adds.

| Mode | Collective it adds | How often | Bytes per rank | Slowest link it works on |
|---|---|---|---|---|
| DDP | all-reduce of gradients | once per backward pass | `2 × (R−1)/R × 2P` | network |
| FSDP | all-gather and reduce-scatter | once per layer group per backward pass | `2 × (R−1)/R × 2P`, or 3 × under the ZeRO-3 analog | network, at a cost |
| hybrid sharding | the same inside a domain, then an all-reduce across domains | per backward pass | across domains, only the shard each rank kept | network |
| tensor | all-reduce of activations | 2 per layer forward, 2 backward | `4 × 2 × (R−1)/R × 2TH` per layer | NVLink |
| expert | all-to-all of routed tokens | 4 per MoE layer per microbatch | in proportion to routed tokens × hidden size | NVLink; the network at 0.22 of the rate |
| context | all-to-all of activations | 2 per attention layer forward, 2 backward | in proportion to tokens × hidden size | NVLink |
| pipeline | send and receive | once per boundary per microbatch | `4TH` per step per boundary | network; TCP sockets at 0.875 of the rate |

The clearest way to see what communication costs is to run different layouts on the same hardware. These rows are two 8-GPU B300 nodes over EFA with gradient checkpointing off, one run each, recorded on Halo 1.2.0 ([Pipeline Parallelism → Measured throughput](../parallelism/pipeline-parallelism.md#measured-throughput)):

| Model | Layout | tokens/s/GPU |
|---|---|---|
| Qwen3-8B | `dp16`, 16 replicas | 18,337 |
| | `pp2 × dp8`, 2 pipeline stages of 8 replicas | 17,608 |
| | `dp8 × tp2`, 8 replicas of a 2-way tensor split | 10,780 |
| | `dp16` on TCP sockets | 1,117 |
| | `pp2 × dp8` on TCP sockets | 15,414 |
| gpt-oss-20b | `ep8`, experts over 8 per node, 2 replicas | 18,685 |
| | `pp2 × ep8`, 2 stages with experts over 8 in each | 15,715 |
| | `ep16`, experts over all 16 | 4,115 |

![Bars of tokens/s per GPU on B300 in four groups. Two nodes on a fast link: the pipeline is 4.0% below 16 replicas and 15.9% below experts over 8 per node. The same nodes on TCP sockets: the pipeline holds 15,414 where 16 replicas fall to 1,117. One node: a dense pipeline is 8.5% below 8 replicas, and a pipeline with experts is 9.2% above experts over 8. A larger model: a pipeline runs 1.61 times experts over all 16.](../assets/benchmarks/pp_throughput.png)

Read the table for the three conditions under which communication is the limit.

- The tensor split puts a collective on the forward path of every layer and runs at 0.59 of plain replicas.
- Experts over 16 send a collective across a slow link at every layer and run at 0.22.
- Plain replicas on sockets run everything on a slow link, at 0.06.

> In Halo: data-parallel size = world size / pipeline stages / max(tensor, context, expert-tensor size). It is a maximum where you might expect a product, because only one of the three may be above 1 in a run ([Parallelism → Stacking the dimensions](../parallelism/README.md#stacking-the-dimensions)). The supported combinations are a fixed list ([Parallelism → Supported combinations](../parallelism/README.md#supported-combinations)).

Check yourself:

1. A job has 16 ranks, 2 pipeline stages and a 2-way context split inside each stage. How many replicas does it have?
2. Which modes divide the activations a rank holds, and which divide its weights?

[Answers](#answers-on-combining-the-modes) are at the end of the page.

### Overlapping communication with compute

A collective takes little or no time from your step if it runs while the GPU computes something else. That works when three things hold:

- The next operation does not need the collective's result.
- Its kernel needs few blocks, at most 64, and they find places on the SMs as compute blocks finish.
- The CPU thread is free to launch both.

It works for the gradient buckets of DDP and for the reduce-scatter of FSDP, which runs beside the backward pass of the next layer group. Deferring those reductions gains only a few percent over NVLink, because they were already overlapped ([Data Parallelism](../parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync)).

It does not work for a plain all-reduce of tensor parallelism, whose sum the next operation consumes, or for a pipeline boundary under 1F1B.

Some MoE layers also apply a few shared experts to every token. Running those beside the token exchange is marginal: it gained 2.1 to 2.5% on two models and lost 4.6% on a larger one, and its page does not say why ([DeepEP → Dispatch cost](../infrastructure/deepep.md#dispatch-cost)).

And there is little left to overlap in the token exchange itself, since the dispatch is 9 to 13% of the three phases of an MoE layer.

On one GPU, the measurements in [Streams](#streams) showed that two large kernels on two streams finish no sooner than on one, under the one-queue setting of Halo's image. A GEMM running beside a collective was not measured.

Megatron hides its tensor-parallel collectives under layer compute by another route. With its sequence parallelism the all-reduce becomes a reduce-scatter and an all-gather, and the overlap depends on that and on NVIDIA's Transformer Engine.

> In Halo: the images set `CUDA_DEVICE_MAX_CONNECTIONS=1`, which serializes how one process submits work to its GPU. The expert-parallel suites are validated with it, at no measurable cost in throughput ([Configuration Reference → Environment variables](configuration-reference.md#environment-variables)).

### Why N GPUs are not N times faster

There are two ways to add GPUs. **Strong scaling** keeps the work fixed and spreads it over more GPUs. **Weak scaling** grows the work with the GPUs. Two classical laws predict what each gives.

Let `f` be the fraction of a step that speeds up with more GPUs and `R` the number of GPUs.

```text
Amdahl, strong scaling:    speedup = 1 / ((1 − f) + f / R)
Gustafson, weak scaling:   speedup = (1 − f) + f × R
```

With `f = 0.95`, Amdahl gives 5.9 on 8 GPUs and 15.4 on 64, and never more than 20. Gustafson gives 7.65 on 8 and 60.85 on 64 (computed). The two laws measure `f` on different runs: Amdahl on the run with one GPU, Gustafson on the run with `R`.

![Two plots of speedup over one GPU against the number of GPUs, computed for a parallel fraction of 0.95. To 64 GPUs the ideal line reaches 64, Gustafson's law 60.85, and Amdahl's law bends flat to 15.4 under a limit line at 20. An enlargement of the first 16 GPUs adds four recorded Qwen3-8B points at 1, 6.7, 11.5 and 0.7.](../assets/diagrams/scaling_laws.png)

For training, write the time of a step on one rank as four terms:

```text
step time = fixed part + compute for the rank's tokens + communication + waiting for the slowest rank
```

Strong scaling shrinks the tokens per rank. The fixed part and the communication do not shrink with them. Each GEMM's `M` also falls toward the region where its time stops depending on `M`, which [Why a small M is slow](#why-a-small-m-is-slow) measured, so compute time falls more slowly than the tokens do. Weak scaling keeps the tokens per rank, and only communication and waiting grow. The laws therefore predict that weak scaling holds up better.

Recorded rates of Qwen3-8B give a sense of size. They are on different shapes, so read them as indications ([Throughput Benchmarks](../optimization/throughput-benchmarks.md#single-gpu-dense-1-b300), [Pipeline Parallelism](../parallelism/pipeline-parallelism.md#measured-throughput)).

| Run | tokens/s/GPU | Share of one GPU's rate | Speedup |
|---|---|---|---|
| one B300 at its best shape | 25,569 | 1 | 1 |
| 8 replicas on one node | 21,359 | 0.84 | 6.7 |
| 16 replicas on two nodes over EFA | 18,337 | 0.72 | 11.5 |
| 16 replicas on two nodes on TCP sockets | 1,117 | 0.04 | 0.7 |

Every GPU of those runs brings its own batch, which is weak scaling, so hold the points against Gustafson's line and not against Amdahl's curve.

The `M` per GPU shrinks only in two cases. One is when the added GPUs share a batch, as under tensor parallelism, context parallelism and pipeline microbatches. The other is when you lower the batch per GPU to hold the total fixed.

### How distributed runs fail

Most failures of a multi-GPU run fall into seven classes. Each has its own mechanism, and several of them look the same from outside.

1. A collective that not every rank calls. One rank enters it and another never reaches it, so the job blocks until a watchdog ends it, 10 minutes later by PyTorch's default.
2. Different inputs where the same are required. Tensor-parallel ranks that were given batches of different shapes stop at the first all-reduce. With equal shapes and different contents this is class 7.
3. A slow rank. The run is slow and correct.
4. A compile or a load on one rank. That rank arrives late at the next collective, and a late rank alone only delays its peers. It becomes a deadlock when the group falls out of step. In one recorded case the peers went ahead into the next collective while a rank compiled on a cache miss, and the ranks ended up waiting for each other ([Flash Attention → Known issues](../optimization/flash-attention.md#known-issues)).
5. An error on one rank. That rank raises. Its peers block in the next collective and are killed later by the watchdog, with a message that blames the wrong line. The remedy is to agree across ranks first and then raise on all of them.
6. Out of memory on one rank. Memory is not equal across ranks: one rank has the most routed rows, one writes the checkpoint, one got the longer row. For the other ranks this is class 5.
7. Silent divergence, with no hang and wrong training. The replicas never had the same initial state, or a gradient was averaged where a sum was due. Or a loss normalizer does not cancel, so that every gradient comes out too small by a constant factor. Or an environment variable differs between nodes, or ranks that share a batch were given different batches of one shape.

![Two rows of cells, R0 and R1, in call order. Calls 1 to 5 are the same on both: three all-gathers, then two reduce-scatters. At call 6 R0 issues the reduce-scatter of group 1 while R1, which never issued it, issues the next step's all-gather. Both cells are marked, and each rank then waits in a dashed stretch until the watchdog fires after 10 minutes.](../assets/diagrams/collective_mismatch.png)

The figure draws the usual cause of the first class, a difference between ranks in the autograd graph. If the output of a layer group is left disconnected from the loss on one rank, which a batch of fully masked rows can do, autograd prunes that group's backward pass there and its reduce-scatter never runs.

In a hung job every GPU whose rank is waiting inside a collective reads 100% utilization at the power of an idle process. A GPU that reads 0% belongs to a rank that is somewhere else, in the data loader, in another branch or already dead, and that is the rank to read first. Dump the stack of every rank and compare them.

If every GPU reads 100%, every rank is in a collective, and they are in different ones, because one that every rank had entered would finish. PyTorch's flight recorder says which. It is a ring buffer of each rank's recent collectives, dumped when the watchdog fires if you set `TORCH_NCCL_DUMP_ON_TIMEOUT=true` before the run. At one sequence number it shows one rank calling a different collective from its peers ([Debugging → Diagnosing multi-node hangs](debugging.md#4-diagnosing-multi-node-hangs)).

Read the stacks with the launch model in mind. A collective is queued like any kernel and does not block the CPU. A blocked rank's stack therefore shows the first line where the CPU needed a result from the GPU, which can be later than the collective that is stuck.

> In Halo: the watchdog is set to 30 minutes. The token exchange has its own budget of 100 s for how far ranks may drift apart, separate from the NCCL watchdog ([DeepEP → Environment variables](../infrastructure/deepep.md#environment-variables)). The symptom tables are on [Troubleshooting](troubleshooting.md).

Check yourself:

1. 90% of a step speeds up with more GPUs. What speedup does Amdahl's law give on 16 GPUs at fixed work, and Gustafson's law with the work grown 16 times?
2. Rank 3 of a job raises an out-of-memory error. Ten minutes later the other ranks die with a watchdog message that names a collective. Whose log do you read first, and which class is this for the other ranks?
3. Every GPU of a hung job reads 100% utilization at 238 W. What does that tell you, and what do you do next?

[Answers](#answers-on-how-distributed-runs-fail) are at the end of the page.

## Around the training step

### The host and the input pipeline

One CPU thread at a time launches the kernels of a rank and runs every hook of the framework. The forward pass uses the thread of your training loop, and the backward pass a thread of the autograd engine. When that thread is the limit, the whole step is overhead-bound, whatever its kernels are. "Host" in names like host-bound means this CPU side of a rank.

The evidence is a set of changes that mainly removed CPU work, each recorded in Halo's benchmarks:

| Change | Gain |
|---|---|
| about 31 small operations per MoE layer pass replaced by three kernel launches | +2.2% to +10.6% across five layouts, four of them gpt-oss-20b ([Grouped GEMM](../optimization/grouped-gemm.md#performance)) |
| two decoder layers per FSDP group in place of one, which halves the collectives, the copy launches and the hook calls | −0.5% to +11.3% across seven layouts of gpt-oss-20b and Qwen3-8B ([Data Parallelism](../parallelism/data-parallelism.md#shard-groups)) |
| plain linear layers on each rank in place of a wrapper that made about ten extra calls per linear layer | +57.5% on Qwen3-8B under a 2-way tensor split ([Tensor Parallelism](../parallelism/tensor-parallelism.md#local-shard-linears)) |
| sizing a kernel's work from counts the CPU already has, so that no value is read back from the GPU | +6.6% on one layout ([Grouped GEMM](../optimization/grouped-gemm.md#performance)) |
| binding each process to the CPU cores nearest its GPU | +3.4% on one overhead-bound layout and +1.4% on another, nothing on layouts that are not overhead-bound ([Throughput Benchmarks](../optimization/throughput-benchmarks.md#numa-binding)) |

Halo's hyper-connection kernels are another case. They compute the residual mixing of two model families with one launch per direction, and turn 143 forward and 306 backward launches into 16 and 14 ([Hyper-connection kernels](../optimization/hyper-connection-kernels.md)).

The input itself is small for text. A row of 8,192 token ids as 64-bit integers is 64 KiB (computed). The cost is in tokenizing and collating on the CPU, and in whether the copy to the GPU makes the CPU wait.

That copy has two settings. PyTorch's notes say that "host to GPU copies are much faster when they originate from pinned (page-locked) memory". Measured on the B300 with 1 and 8 GiB buffers, a copy from pinned memory to the GPU runs at 56 GB/s and one from ordinary pageable memory at 9 GB/s. In the other direction the rates are 57 and 17 to 19 GB/s.

A copy from pinned memory can also be non-blocking, which lets it run behind the queued kernels. In Halo's benchmarks that gained 11.4% on Qwen3-8B with context parallelism over 2 ranks and nothing under plain FSDP ([Configuration Reference](configuration-reference.md)).

Two measurements tell you whether the host is your limit. One is the time the training loop waits for its next batch, which should be near zero. The other is a flame graph of the Python process ([Debugging → py-spy](debugging.md#1c-py-spy--cpu-flame-graph-dataloader--python-stalls)). If the loop does wait for data, tokenize ahead of time, add loader workers, or let the loader prepare more batches ahead.

Keep the CPU cores apart as well. Compile workers, data-loader workers and the thread that launches kernels should not share cores.

The link between CPU and GPU matters more for images and video, for loading weights from CPU memory and for any transport that copies through the CPU. On a B300 node it is the PCIe link in the table of [The wires](#the-wires).

> In Halo: with the efficiency callback on, the wait for the next batch is logged as `step_data_wait_seconds` and `avg_data_wait_seconds` ([Callbacks → EfficiencyCallback](../training-methods/callbacks.md#efficiencycallback)).

### Checkpoints and storage

A job can die during a save that was making progress. Saving a model that is split across ranks is a collective with one writer: every rank takes part, one rank writes, and the rest wait. A slow writer can outlast the watchdog described in [How distributed runs fail](#how-distributed-runs-fail).

A saved checkpoint of a model is 2 bytes per parameter of bf16 weights. If you want to resume from it, add 4 bytes per parameter of optimizer state. For the 117B parameters of gpt-oss-120b the weights alone are 234 GB (computed).

The time of a save is the bytes divided by the slowest of three rates: the gather between GPUs, the copy from the GPU to CPU memory, and the disk or network filesystem behind the writer. For those 234 GB the copy alone takes 4 s at the pinned rate of 57 GB/s and 12 to 14 s at the pageable rate of 17 to 19 GB/s (computed).

A sharded save lets every rank write its own piece in parallel. It exists for write speed and saves no memory.

Loading is the mirror: one reader per rank, all reading the same files. When the ranks of a node read the same file, the operating system reads it from disk once and serves the others from its page cache.

> In Halo: an expert-parallel gathered save on a shared filesystem spreads its layers over one writer per node by default; the other gathered saves have a single writer (each node's local rank 0 on per-node storage). The layouts on disk, what a resume needs and the save paths per parallel mode are on [Checkpoints](checkpoints.md).

### Training and inference

You may have read that fp8 made inference twice as fast, and concluded that it will speed up training. The two workloads hit different limits, and the roofline shows where they part.

Inference has two phases. **Prefill** runs all tokens of the prompt at once. It has a large `M` and behaves like the forward pass of training. **Decode** produces one new token per running sequence per step, so the `M` of a decode GEMM is the number of sequences that are decoding together.

#### Three decode regimes

With one sequence or a few, decode is the small-`M` case at its smallest. At `M = 1`, a 4096 × 4096 bf16 layer has an AI of 4,096 / 4,098 = 1.0 FLOP/byte (computed), the same as an elementwise kernel. These times were measured on one B300, with the weight read from HBM:

| Rows, the sequences decoded together | AI | Device time | Rate |
|---|---|---|---|
| 1 | 1.0 | 8.66 µs | 3.9 TFLOP/s |
| 8 | 8.0 | 8.99 µs | 29.9 TFLOP/s |
| 16 | 15.9 | 12.04 µs | 44.6 TFLOP/s |
| 32 | 31.5 | 8.46 µs | 126.8 TFLOP/s |
| 128 | 120.5 | 8.89 µs | 483.4 TFLOP/s |
| 256 | 227.6 | 11.88 µs | 723.2 TFLOP/s |
| 512 | 409.6 | 16.77 µs | 1,024.6 TFLOP/s |
| 2,048 | 1,024.0 | 53.70 µs | 1,279.6 TFLOP/s |

The GEMM takes 8.5 to 9.2 µs at every row count measured from 1 to 128 except 16, where cuBLAS picks a slower kernel and the call takes 12.0 µs. At 1 row that is 3.9 TFLOP/s, 0.3% of the sustained GEMM peak. Decoding 128 sequences together does 128 times the work in the same time. The flat time is the one [Why a small M is slow](#why-a-small-m-is-slow) fitted. This weight is one of its four shapes, and 3.9 µs + 33.6 MB / 7.7 TB/s gives 8.3 µs.

With few sequences the CPU sets the pace. In a second run, without the profiler, the GEMM with 1 to 8 rows took 11.5 µs per call from plain PyTorch against 8.7 to 8.8 µs on the device. At those row counts cuBLAS runs two kernels per call, one that splits each tile along `K` and one that adds the parts. Inference engines replay the decode step from a CUDA graph to take the CPU out of that path.

![Two panels of lines against the rows M of one bf16 layer, [M, 4096] @ [4096, 4096], from 1 to 2,048 on a log axis. Above, the time per call on the device is flat near 9 µs from 1 row to 128, except 12 µs at 16, and reaches 54 µs at 2,048. A second line, calls from PyTorch, sits at 11.5 µs up to 8 rows and then runs just above the first. Below, the rate on the device rises from 3.9 to 1,280 TFLOP/s under the memory line and the 1,388 roof, with the crossover M at 232.](../assets/diagrams/decode_roofline.png)

With many sequences decode leaves the flat stretch. The time starts to grow between 128 and 256 rows, where the AI passes the ridge. At 2,048 rows the layer is compute-bound and runs under the power cap.

The third regime is the engine itself. An engine whose core loop is one Python thread runs out of CPU before it runs out of GPU. Halo's notes on its two serving engines say that "decode is CPU-bound at RL concurrency", with one core saturated from a few dozen concurrent sequences ([Rollout Servers → Throughput](../infrastructure/rollout-servers.md#throughput)). In the terms of this lecture that is overhead-bound.

#### Why low precision helps one and not the other

In the first decode regime the time grows with the bytes of the weight, about half of it by the fit above, and a quantized weight has fewer bytes: half for fp8 and a quarter for fp4. With their block scales the exact figures are 8.25 bits per weight for mxfp8, 4.25 for mxfp4 and 4.5 for nvfp4 (computed). No quantized decode GEMM was timed for this lecture.

Decode has a second large object in memory, the **KV cache**. It holds the keys and values of past tokens, which decode keeps so that it does not compute them again. Its size per token is 2 × full-attention layers × key-value heads × head width × 2 bytes. Qwen3.6-35B-A3B has 10 such layers among its 40, with 2 key-value heads of width 256, which makes 20 KB per token (computed). Weight quantization does not touch it, and engines quantize it with a separate setting.

Training does not get the same benefit, for three reasons. The master weights stay in bf16 or fp32, so no bytes are saved. A quantization pass is added to every step. And the backward pass adds two GEMMs per layer that the scheme has to cover too.

| | Training | Decode |
|---|---|---|
| GEMMs per layer | forward, dgrad and wgrad | forward only |
| Largest memory | weights, gradients and optimizer state at 8 bytes per parameter, and activations | weights and the KV cache |
| Rows per GEMM | the tokens of a rank; per expert, a `k/E` share of them | the running sequences |
| Usual regime | compute-bound at large `M`; split runs add collectives and waiting | memory-bound at low concurrency; overhead-bound in a Python engine at high concurrency |
| Low precision | 1.13 to 1.37 times in the large runs cited under [Precision](#when-low-precision-speeds-training-up); below 1 in the whole steps Halo has measured, the native ones on the 1.2.0 development build's adapter | less memory always; faster on some engines |
| What to tune | a larger `M`, fusion, checkpointing, overlap | weight quantization, the KV cache, batching |

Even for serving, the speed gain is not a given. On Qwen3.5-35B-A3B with fp8 weights that carry one scale per 128 × 128 block, decode on vLLM stayed within 1% of bf16 on one server. On SGLang with its Triton MoE backend it was 26% faster, at 256 sequences of 1,024 tokens on one B300. What fp8 gave on both engines was memory ([Rollout Servers → FP8 rollouts](../infrastructure/rollout-servers.md#fp8-rollouts)).

#### An RL step does both

An on-policy reinforcement-learning step sends the current weights to a generator, generates samples with them, scores the samples and trains on them. Generation is decode and training is a training step. No one configuration suits both, so they run on separate GPUs: an inference engine on some and the trainer on the rest.

The engine does what decode needs. It keeps a KV cache, replays the decode step from CUDA graphs and uses continuous batching, in which sequences join and leave the running batch at every step.

Such a step is usually limited by generation. If the step is serial, transfer and then generate and then train, the trainer's GPUs sit at the power of an idle process while the engine generates, and that is expected. If the next round is generated while this one trains, the step time becomes the transfer plus the longer of generation and training.

Three changes act on that time.

- Generate the next round while training on this one. The samples then come from weights that are one or more rounds old, and the training objective has to correct for that.
- Generate a larger batch. Total throughput rises by less than the batch does.
- Transfer the weights less often. Every transfer pauses every engine, and fewer transfers save that pause more than they save bytes.

How long the transfer takes follows from the link rates in [The wires](#the-wires). A full model is 41.8 GB for gpt-oss-20b and 234 GB for gpt-oss-120b (computed: parameters × 2 bytes). Inside a node it goes from GPU to GPU over NVLink. Across nodes, Halo recorded 53 and 80 GB/s through its two engine clients over EFA and 9.7 GB/s on sockets ([Rollout Servers](../infrastructure/rollout-servers.md#servers-on-other-nodes-efa)). At those rates 41.8 GB takes 0.8 s, 0.5 s or 4.3 s on the wire (computed).

The whole transfer takes longer than the wire. In one timed push of Qwen3.5-35B-A3B in bf16 to one vLLM server, 0.9 s of the 1.7 s went to the engine's own loader ([Rollout Servers → FP8 rollouts](../infrastructure/rollout-servers.md#fp8-rollouts)).

One fact will save you a long search. In bf16, a trainer and an engine do not produce identical log-probabilities from identical weights, and the difference is not a transfer error ([Trainer and engine noise](../training-methods/async-grpo/objective.md#trainer-and-engine-noise)).

> In Halo:
>
> - The engines are separate programs with their own software stacks, and the weight transfer needs a distinct GPU for every participant ([Servers and Launch](../training-methods/async-grpo/setup.md#gpu-split)).
> - The transfer writes weights in place, so an engine has to store them in the layout the trainer sends. It runs at the start of a step, before that step's generation.
> - Engines serve bf16 by default. An fp8 option exists at a cost: a timed push of 5.9 s against 1.7 s, and a wider gap between the trainer's and the engine's log-probabilities, 0.064 against 0.042 nats on average ([Rollout Servers → FP8 rollouts](../infrastructure/rollout-servers.md#fp8-rollouts)).
> - The three changes above are settings of the environment-based method ([Memory and Throughput → Throughput levers](../training-methods/async-grpo/performance.md#throughput-levers)).

Check yourself:

1. One 4096 × 4096 bf16 layer decodes 1 sequence in 8.7 µs on the device. What do you expect for 64 sequences decoded together, and what does that do to the rate?
2. A model has 24 full-attention layers with 8 key-value heads of width 64. How large is its KV cache per token in bf16, and for 100 sequences of 8,000 tokens?
3. How long do the 234 GB of gpt-oss-120b take on the wire at 53 GB/s, and why is the whole transfer longer?

[Answers](#answers-on-training-and-inference) are at the end of the page.

## Finding the bottleneck

### A method for one step

The opening promised a way to find which limit your step is hitting. It is seven checks, in this order.

1. Measure the rate. Count real tokens/s per GPU over steady steps. Skip the first few steps, wait for the GPU before you read the clock, and repeat. Small-batch MoE runs vary by about 10% from run to run ([PEFT → Measured cost](../optimization/peft.md#measured-cost)).
2. Compute what the step could do. Count `6 × P` FLOPs per token for trained parameters and `4 × P` for frozen ones, and add attention. A peak divided by that count is a rate in tokens/s. Divide the sustained GEMM peak, 1,388 TFLOP/s on this board, and you have the rate a step can approach. MFU, which the next section defines, divides by the reference peak of 2,250 instead. For Qwen3-8B the two rates are 28,200 and 45,800 tokens/s (computed from `6 × P` alone). For a mixture-of-experts model count the parameters a token uses, its `k` experts of `E` and everything outside the experts. That is 3.6B of 20.9B for gpt-oss-20b, which makes its two rates about 64,000 and 104,000 tokens/s (computed).
3. Check that the GPU is fed. The time the loop waits for its next batch should be near zero, as [The host and the input pipeline](#the-host-and-the-input-pipeline) describes. In an RL step this wait is generation, and [An RL step does both](#an-rl-step-does-both) has the three changes that act on it.
4. Read power, utilization and the SM clock together over a few seconds of samples, as in [Power and utilization](#power-and-utilization). A low reading shows a step paced by the CPU or a rank that only waits. A high reading rules nothing out on a whole step.
5. On several GPUs, compare the ranks with each other before you profile one of them. A rank's time inside a collective includes its wait for the slowest rank, so it is not time on the wire. Compare the tokens each rank processed and its time in each phase. The rank whose time inside a collective is shortest is the one the others wait for, and if that spread is large, the collective's time is waiting and not transfer. The wait shows up in whichever operation first makes the CPU wait for the GPU, which is often the `.item()` on the loss.
6. Split the time of the step with a profile and sort the kernels by GPU time. A profiler slows the step it measures, so read shares from it and rates from a run without it. For each of the top few kernels, compute what it should cost: a GEMM from its FLOPs and bytes, an elementwise kernel from its bytes. Then read the trace for four marks.
   - A kernel well above its prediction is the one to look at.
   - Thousands of launches of a few microseconds each mean an overhead-bound step.
   - Gaps on the stream under a collective are communication or waiting.
   - Gaps with a busy CPU thread are host work.
7. Change one thing and measure again, over about 10 steady steps.

PyTorch's profiler does the sixth check in a few lines. Put your own step in place of the one here:

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = torch.nn.Linear(4096, 4096, device="cuda", dtype=torch.bfloat16)
x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)

def step():  # your forward pass, backward pass and optimizer step go here
    model(x).float().square().mean().backward()

for _ in range(3):  # warm-up, so no first call is in the profile
    step()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

On one B300 it printed 15 rows. These are six of them, with the CPU columns cut:

```text
Name                                                  Self CUDA   CUDA total   # of Calls
aten::addmm                                             1.483ms      1.632ms           10
aten::mm                                                1.616ms      1.616ms           10
nvjet_sm103_tst_128x256_64x6_2x2f_2cta_h_bz_NTT         1.616ms      1.616ms           10
nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_bias_TNT     1.483ms      1.483ms           10
aten::copy_                                             1.622ms      1.622ms           30
aten::mul                                               1.006ms      1.006ms           20
Self CUDA time total: 7.620ms
```

Each row is an operation with its time on the GPU and its number of calls. The two `nvjet` rows are cuBLAS GEMM kernels, named as in [Tiles and kernel selection](#tiles-and-kernel-selection): the one with `bias` in its name is the forward pass and the other is the weight gradient. Together they take 41% of the GPU time, at 0.15 and 0.16 ms per call. Divide time by calls in this way to see whether a kernel is large or one of thousands of small ones.

Take the Qwen3-4B step of [Power and utilization](#power-and-utilization) through the first five checks. It is a full fine-tuning step on one B300, on one row of 8,192 tokens with gradient checkpointing off and PyTorch's own kernels.

1. The rate is 8,192 tokens in 381.6 ms, about 21,500 tokens/s (measured).
2. `6 × P` is 6 × 4.02 × 10⁹ = 2.41 × 10¹⁰ FLOPs per token. Attention at this row length adds 7 × 8,192 × 4,096 × 36 = 8.5 × 10⁹ by the causal count, where 4,096 is this model's 32 heads of width 128. The step therefore does about 700 TFLOP/s, which is 50% of the sustained GEMM peak and 31% of the MFU reference peak (computed). There is room.
3. The run fed one fixed batch of token ids from GPU memory, so it never waited for data.
4. Power is 95% of the power limit and utilization 98%, and the clock averages 1,927 MHz with the power cap at times. At 95% of the power limit little of this step can be waiting or paced by the CPU.
5. It has one GPU, so there are no ranks to compare.

No kernel profile of this step was taken. The run with fused kernels in [Where fused kernels come from](#where-fused-kernels-come-from) stands in for the sixth check: the forward and backward passes of the same model got 25 to 34% shorter when four kinds of memory-bound kernel were fused, so a good part of the step is memory-bound work.

Halo's own profiles show the last two checks done in full. On a gpt-oss-20b step with its experts split across GPUs, one `index_add_` kernel takes about 20% of all GPU time. A gather and a reduction in its place run about 5 times faster, and board power rises from about 34% to about 56% of the power limit ([Grouped GEMM](../optimization/grouped-gemm.md#the-duplicate-index-gather-trap)). In another profile an attention backward pass takes 22% of kernel time, and it is absent from the profile with a block-sparse kernel in its place ([Flash Attention](../optimization/flash-attention.md#block-sparse-attention-qwen38-flash-next)).

> In Halo: the profiler setting writes a trace and a list of the 25 operations with the most GPU time, and it names the phases of the token exchange. The trace does not complete on a multi-GPU run with experts split and FA4 active ([Debugging](debugging.md#1-flame-graphs)). The efficiency callback computes rate, step time, memory and MFU, and MFU reaches the log only when the MFU diagnostics are on as well (`report_mfu_diagnostics`).

### MFU

**MFU**, model FLOPs utilization, is the FLOPs per second that the model's forward and backward passes require, divided by a peak. To first order:

```text
MFU ≈ 6 × P × T / (peak × step time)
```

The first question about any MFU figure is which peak it divides by. The figures quoted in this lecture divide by the MFU reference peak, 2,250 TFLOP/s for bf16 on a B300. A step that ran as fast as a sustained GEMM would read 1,388 / 2,250 = 0.62, or 0.65 at the best shape measured. At the burst rate it would read 0.80 to 0.84 (computed).

An MFU of 1.0 is not reachable on this board. Divide a measured rate by 148 SMs and by the SM clock and you get FLOPs per SM per clock cycle, which you can hold against the 8,192 of the model behind the [ceiling at the maximum clock](#two-more-numbers-called-the-peak):

| GEMM loop | SM clock | Rate | Share of 8,192 FLOPs per SM per cycle |
|---|---|---|---|
| sustained, under the power cap | 1,185 to 1,260 MHz | 1,388 to 1,468 TFLOP/s | 96 to 97% |
| first 30 ms from a cool start | 2,006 to 2,011 MHz, read by a probe kernel between chunks of GEMMs | 1,796 to 1,809 TFLOP/s | 73 to 75% |

The reference peak of 2,250 TFLOP/s is what 148 × 8,192 FLOPs per cycle give at a clock of 1,856 MHz. Under a sustained GEMM the 1,100 W limit of this board holds the clock near 1,200 MHz, and there the kernel already does 96 to 97% of the model's figure. At the full clock the same kernel does about three quarters of it, and what holds it there was not measured.

MFU leaves recomputation out. The definition in the PaLM paper counts "the required operations to compute the forward+backward passes, and not rematerialization". The figure that includes it is hardware FLOPs utilization, HFU. PaLM reports 46.2% MFU and 57.8% HFU.

So turning gradient checkpointing on lowers MFU while the hardware does as much work as before. Qwen3-8B on one B300 at 4 rows of 4,096 tokens, forward and backward pass only, ran 17,150 tokens/s without checkpointing and 13,157 with it. By the `6 × P` count those are 843 and 647 TFLOP/s, an MFU of 37.5% and 28.7% (measured rates, computed MFU).

For comparison, the PaLM paper lists 30.2% for Megatron-Turing NLG and 21.3% for GPT-3 beside its own 46.2%. In Halo's records a dense single-GPU step on a B300 reads 60% on Qwen3-8B and 51% on Qwen3-4B, both at 16 rows of 2,048 tokens without gradient checkpointing ([Throughput Benchmarks](../optimization/throughput-benchmarks.md#single-gpu-dense-1-b300)).

60% of 2,250 is 1,350 TFLOP/s by this count, close to the sustained GEMM peak. Halo's count credits the embedding as if it were a matmul, which alone makes it about 7% high on Qwen3-8B, and it adds attention at every key (the note below gives the numerator).

The 60% and the 37.5% above are the same model. The 37.5% is a forward and a backward pass with PyTorch's own norms and loss at 4 rows of 4,096 tokens. The 60% is a whole step with fused kernels at another shape.

MoE models need a correction. The `6 × P` rule assumes that every parameter works on every token. In an MoE layer each token uses its `k` experts of `E`, so gpt-oss-20b uses 3.6B of its 20.9B parameters per token. Plain MFU then counts FLOPs that did not run. A sparsity-aware MFU scales the expert term by the share of experts a token uses.

The two metrics answer different questions.

- Tokens/s per GPU compares layouts of one model on the same rows, and it tells you which GPU is faster. It does not hold across a change of model or of row length. On Qwen3-4B on one B300, without gradient checkpointing, the same 32,768 tokens ran at 41,022 tokens/s as 16 rows of 2,048, at 38,936 as 8 rows of 4,096 and at 34,775 as 4 rows of 8,192 ([Throughput Benchmarks](../optimization/throughput-benchmarks.md#single-gpu-dense-1-b300)).
- MFU compares how well different GPUs are used. 40% of a B300 is 900 TFLOP/s and 50% of an H100 is 495 (computed from the two reference peaks, 2,250 and 989.5 TFLOP/s). It does not tell you which run is faster.

> In Halo:
>
> - The numerator is finer than `6 × P × T`. It is tokens × (6 × trainable + 4 × frozen parameters) per rank, plus the FLOPs of the attention scores.
> - MFU and the sparsity-aware S-MFU are computed when the efficiency metrics are on. They reach the log only when the MFU diagnostics are on as well (`report_mfu_diagnostics`). If the top-k of a model is not detected, the log carries no S-MFU fields and no `S-MFU:` line at startup ([Throughput Benchmarks → Metrics](../optimization/throughput-benchmarks.md#metrics)).
> - The "achieved TFLOPS" column compares only configurations with the same number of parameters per rank. Going from experts over 2 ranks to experts over 8 lowers tokens/s per GPU by 36% and that column by 75% on gpt-oss-20b at batch 4 ([Throughput Benchmarks → Metrics](../optimization/throughput-benchmarks.md#metrics)).

Check yourself:

1. A step runs 12,000 tokens/s per GPU on Qwen3-8B on a B300. What is its MFU by the `6 × P` count?
2. You turn gradient checkpointing on and MFU falls from 37.5% to 28.7%. Did the hardware get less busy?
3. One run reads 30% MFU on a B300 and another 60% on an H100, whose reference peak is 989.5 TFLOP/s. Which does more model FLOPs per second?

[Answers](#answers-on-mfu) are at the end of the page.

### Reproduce the numbers

Three short programs print three of the numbers this lecture rests on. They need PyTorch with CUDA and nothing else.

The first times a bf16 GEMM and prints its rate every 20 calls. From a cool GPU you should see the burst GEMM peak for the first few lines and the sustained GEMM peak after that.

```python
import time
import torch

n = 8192
a, b, out = (torch.randn(n, n, device="cuda", dtype=torch.bfloat16) for _ in range(3))
torch.mm(a, b, out=out)  # the first call loads the library
torch.cuda.synchronize()
for _ in range(40):
    start = time.perf_counter()
    for _ in range(20):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    print(f"{20 * 2 * n**3 / (time.perf_counter() - start) / 1e12:,.0f} TFLOP/s")
```

The second copies a 2 GiB tensor and prints HBM bandwidth, counting the read and the write.

```python
import time
import torch

src = torch.ones(2**30, device="cuda", dtype=torch.bfloat16)  # 2 GiB
dst = torch.empty_like(src)
dst.copy_(src)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    dst.copy_(src)
torch.cuda.synchronize()
print(f"{100 * 2 * src.nbytes / (time.perf_counter() - start) / 1e12:.2f} TB/s, read plus write")
```

The third times `silu(a) * b` eager and compiled.

```python
import time, torch, torch.nn.functional as F

def time_ms(fn, a, b, calls=200):
    for _ in range(3):  # warm-up; a compiled function compiles on its first call
        fn(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / calls * 1e3

f = lambda a, b: F.silu(a) * b
a, b = (torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16) for _ in range(2))
print(f"eager {time_ms(f, a, b):.3f} ms, compiled {time_ms(torch.compile(f), a, b):.3f} ms")
```

On a B300 these printed 1,777 to 1,803 TFLOP/s for the first four lines and then a median of 1,384, 6.61 TB/s, and 0.116 and 0.063 ms (measured, each program run as printed).

A fourth exercise needs no GPU. Take the largest linear layer of your own model and compute its AI and its crossover M from the formulas in [The roofline](#the-roofline).

Most measurements of this lecture come from eight scripts in Halo's repository, under `tests/gpu/profiling/`. Two more in the same directory gave the low-precision GEMM rates (`benchmark_lowp_kernels.py`) and the loop against the grouped GEMM (`benchmark_grouped_mm.py`). They import Halo's test helpers and Liger, so they run inside Halo's image ([Docker Guide](../infrastructure/docker.md#standard-training-launch)). Each of the eight takes `--output` with the name of a JSON file to write. The first of them runs on one GPU like this:

```bash
docker run --rm --gpus '"device=0"' --ipc=host -v "$(pwd)":/workspace -w /workspace halo:blackwell \
  bash -lc "python tests/gpu/profiling/benchmark_roofline.py --output roofline.json"
```

| Script | What it reproduces |
|---|---|
| `benchmark_roofline.py` | the GEMM rates of [The chip](#the-arithmetic-rate), the launch floor, streams, the sweeps of `M` and the kernel choices, multiples of 8, `silu(a) * b`, and the Triton GEMM against cuBLAS |
| `benchmark_memory_hierarchy.py` | HBM bandwidth by access pattern, memory latency, and copies between CPU and GPU; with `--sections peer_copy` and a second GPU visible, the copy over NVLink |
| `benchmark_attention_scaling.py` | the table of [Measured by row length](#measured-by-row-length) |
| `benchmark_loss_head.py` | the kernel-level table of [The fused loss](#the-fused-loss) |
| `benchmark_training_step.py` | whole steps of a real model: `--experiment activations` for activations and gradient checkpointing, `trainable` for frozen weights and LoRA, `padding` for padding and packing |
| `benchmark_collectives.py` | the collectives by message size, launched with `torchrun`; `--wait_probe 30` holds a rank inside an all-reduce for 30 s, for the waiting row of the power table |
| `benchmark_matmul_precision.py` | the fp32 accumulator and the running bf16 sum |
| `benchmark_first_call.py` | the first, second and third calls of [The first call is slow](#the-first-call-is-slow) |

The collectives script needs one process per GPU:

```bash
torchrun --nproc_per_node=4 tests/gpu/profiling/benchmark_collectives.py --output collectives-4.json
```

Each section of `benchmark_roofline.py` yields one group of numbers, and `--sections` selects them:

| Section | What it measures |
|---|---|
| `peaks` | square GEMM rates by number format; `--cooldown 1.0 --repeat_seconds 0.01` gives the burst rates |
| `sweep` | a sweep of `M` at the weight shapes given as `--sweep_shapes 2880x2880`, eager and on the device; `--kernel_shapes` records the kernel and the grid of each point |
| `alignment` | the multiple-of-8 slowdown; `--alignment_forms wgrad` times the weight-gradient form |
| `launch` | the launch floor and CUDA-graph replay |
| `fusion` | `silu(a) * b` in its variants |
| `streams` | two kernels on one stream and on two |
| `triton_gemm` | a generated Triton GEMM against cuBLAS |
| `memory_bound` | elementwise kernels placed on the roofline |

`--sustain gemm --seconds 20 --output gemm-power.json` runs one kind of work steadily, so that you can read power beside it with the `nvidia-smi` command of [Power and utilization](#power-and-utilization). To place your own model on the roofline, pass your expert or MLP weight shape to `--sweep_shapes` as `KxN`.

### Rules of thumb

These are the rules, in the order you would act on them.

1. Measure real tokens/s per GPU on steady steps first. Then find which limit the step is hitting before you change anything.
2. Read power, utilization and the SM clock together. For a loop of one kind of work on this board:
   - 100% at the 1,100 W limit with a throttled clock is compute-bound.
   - 100% with the full clock and 80 to 94% of the limit is memory-bound.
   - About half utilization at 22 to 34% of the limit is the CPU launching small kernels.
   - 100% at 238 W, the power of an idle process, is a rank that waits.

   A real step mixes these, so a high reading rules nothing out. Compare the ranks and read a profile.
3. Raise `M` until the time of your GEMM grows in step with it. That is several times the crossover M: at a width of 2,880 the kernel runs at 45% of the roof at 256 rows and 88% at 8,192. For elementwise work on large tensors, fuse.
4. Pad tokens cost full GEMM FLOPs, so pack the batch.
5. Keep the widths of a bf16 GEMM multiples of 8. The number of tokens needs no rounding for a plain GEMM.
6. Gradient checkpointing costs about a quarter to a third more time per step. At equal batch, leave it off if the batch fits. Then compare the largest batch that fits with it against the largest without.
7. Train in bf16. Low-precision kernels are faster alone, and whole steps gain 1.13 to 1.37 times in the stacks built around them. Halo's low-precision paths have not beaten bf16 on a whole step.
8. Run every kernel once before you time anything, and before the first collective of a multi-GPU run.
9. A step waits for its slowest rank at every collective, so even out the work per rank before you tune anything else on several GPUs.
10. Adding replicas adds batch. Adding GPUs that share a batch divides it.
11. Under the ZeRO-2 analog of FSDP, which keeps the gathered weights from forward to backward, the weights of a dense model have to fit on one GPU, however many GPUs the job has. PyTorch's `fully_shard` defaults to the other setting, the ZeRO-3 analog.
12. Keep a collective that runs at every layer inside the NVLink domain, and use the smallest expert group that fits, which shortens the wait and also lowers each expert's `M`. A pipeline is for a slow link, for a model whose alternative would send attention or experts across nodes, and for a model that does not fit on one node.
13. Compare layouts of one model in tokens/s per GPU. Use MFU to compare how well different GPUs are used.

## Glossary and notation

The terms are in the order the lecture introduces them, and the symbols follow. Halo's own terms, such as its config names, optimizers and training methods, are in the [Glossary](glossary.md).

| Term | Meaning |
|---|---|
| FLOP | one floating-point multiply or add. |
| Kernel | one function that many GPU threads run at once. |
| GEMM | the kernel that multiplies two matrices, `Y = X @ W`. |
| Row | of a GEMM, one token. Of a batch, one sequence. |
| dgrad, wgrad | the two GEMMs of a linear layer's backward pass, for the gradient of the input and of the weight. |
| SM | streaming multiprocessor, one of the 148 identical units of a B300's chip. |
| Tensor core | the unit of an SM built for matrix multiplication. |
| TF32 | an fp32 matmul run on the tensor cores with inputs rounded to 10 mantissa bits. |
| HBM | high-bandwidth memory, the GPU's main memory. |
| Shared memory | an SM's work area for a kernel, 228 KiB on a B300. |
| Tensor memory | Blackwell's store for the tensor cores' accumulator, 256 KiB per SM. |
| Memory latency | the wait for one load from memory, about 160 ns from HBM. |
| Thread, warp, block, grid | a warp is 32 threads in lockstep, a block is up to 1,024 threads on one SM, and a grid is the blocks of one launch. A block is also called a CTA. |
| Occupancy | the share of an SM's 64 warp slots that a kernel fills. |
| Wave | one set of blocks running at once when a grid has more blocks than the SMs hold. |
| Tail effect | the end of a kernel, when a few blocks still run and the other SMs are idle. |
| Stream | the queue into which the CPU puts kernels for the GPU to run in order. |
| Launch floor | the CPU time to launch one kernel, about 5 to 12 µs depending on the operation and the process. |
| CUDA graph | a recorded sequence of kernels that is replayed with one launch. |
| Arithmetic intensity (AI) | FLOPs per byte moved to and from HBM. |
| Roofline | the line min(peak, AI × bandwidth) against which a kernel's rate is read. |
| Ridge | the AI at which the two parts of the roofline meet, about 210 FLOP/byte. |
| Memory-bound, compute-bound | AI below or above the ridge. |
| Crossover M | the row count at which a GEMM's AI reaches the ridge. |
| Eager | plain PyTorch calls, one after another, with no compilation. |
| Packing | several documents concatenated into one row of a batch. |
| Tile | the rectangle of the output that one block of a plain GEMM kernel computes. |
| Tile quantization, wave quantization | the costs of a partly empty last tile and of a small last wave when the tile size is fixed. |
| cuBLAS | NVIDIA's matrix library, the code behind `a @ b` in PyTorch. |
| MoE | mixture of experts, a layer of `E` expert MLPs of which each token uses `k`. |
| TMA | Tensor Memory Accelerator, the unit that copies tensor pieces between HBM and shared memory. |
| MMA | matrix multiply-accumulate, one tensor-core instruction. |
| Stage, of a kernel | one of the shared-memory buffers that let a kernel load ahead of its multiply. |
| Warp specialization | some warps of a block only start loads and the others only compute. |
| Epilogue | the end of a GEMM kernel, where the accumulator is written out. |
| CUTLASS, CuTe DSL | NVIDIA's open-source kernel templates in C++, and the Python kernel language that comes with them. |
| Cluster | a group of blocks that the GPU runs at the same time inside one group of SMs. |
| Overhead-bound | paced by the cost each kernel carries whatever its size, on the CPU or on the GPU. |
| Grouped GEMM | many matmuls with different row counts in one kernel. |
| Fusion | several operations in one kernel, with intermediates kept in registers. |
| Triton, Liger | a Python language for GPU kernels, and a library of fused kernels written in it. |
| Logits | the model's output scores, one per vocabulary entry per position. |
| Fused linear cross-entropy | a loss computed chunk by chunk without ever building the full logits. |
| Flash Attention | attention computed tile by tile without storing the score matrix. FA2, FA3 and FA4 are its kernels for three GPU generations. |
| SDPA, cuDNN | PyTorch's `scaled_dot_product_attention`, and NVIDIA's library of neural-network kernels that it can call. |
| Autocast | mixed precision with fp32 parameters and inputs cast down per operation. |
| ULP | unit in the last place, the gap between neighboring values of a format. |
| Stochastic rounding | rounding up with a probability equal to the dropped fraction. |
| First moment, second moment | AdamW's running averages of the gradient and of its square. |
| Master weights | an fp32 copy of the weights kept beside a lower-precision working copy. |
| Block scaling | one stored scale per small block of low-precision values. |
| Activations | the tensors of the forward pass that the backward pass needs. |
| Allocated, reserved | memory in tensors, and memory the allocator holds. |
| Fragmentation | reserved memory that is free and has the wrong size for the next request. |
| Gradient checkpointing | keeping one input per layer and running the layer's forward again in the backward pass. |
| Reentrant | the kind of gradient checkpointing that runs the forward under `no_grad` and runs it again, with gradients, inside the backward pass. |
| Megatron-LM, selective recomputation | NVIDIA's training system for large language models, and its checkpointing of the attention operations only. |
| LoRA | training small added matrices beside frozen weights. |
| Rank, node, world size | one process with its GPU, one server, and the number of ranks. |
| Replica | one of several copies of the model that each run their own batch. |
| Process group | a subset of ranks with its own connections and buffers. |
| Collective | one coordinated transfer among the ranks of a group. The four kinds are all-gather, reduce-scatter, all-reduce and all-to-all. |
| Waiting | the time a rank is idle until a slower rank reaches a collective. |
| NCCL | NVIDIA's collective library. |
| Bus bandwidth | NCCL's measure of a collective's rate, comparable with a link's rate. |
| EFA | the network adapter of AWS's GPU nodes. |
| NVLink domain | the GPUs that reach each other over NVLink. |
| NVSwitch, NVLS | the NVLink switch chip, and its ability to sum an all-reduce. |
| RDMA | a network card reading and writing memory without the operating system in the path. GPUDirect RDMA extends it to GPU memory. |
| Data parallelism | every rank holds the same model and runs a different batch. |
| DDP, FSDP | data parallelism with whole copies, and with the model stored in shards. |
| Shard | the `1/R` piece of a parameter, a gradient or optimizer state that one rank stores under FSDP. |
| Layer group | the few consecutive layers that FSDP gathers and frees together. |
| ZeRO | DeepSpeed's levels of sharding, used here to name FSDP's two options. |
| Hybrid sharding | shards inside an NVLink domain and replicas across domains. |
| Microbatch | one forward and backward pass of several within an optimizer step. |
| Tensor parallelism | the weight matrices split by columns and rows across a group that shares one batch. |
| Expert parallelism | the experts of an MoE layer spread over a group whose ranks each run their own batch. |
| Dispatch, combine, token exchange | the two all-to-alls of expert parallelism, and the pair of them. |
| Expert-tensor parallelism | the split of tensor parallelism applied inside each expert. |
| Context parallelism | one long row split across a group, with all-to-alls around attention. |
| Pipeline parallelism | the layers split into consecutive stages that microbatches flow through. |
| Stage, of a pipeline | a run of consecutive layers on one group of ranks. |
| Bubble | the share of a step that the stages of a pipeline are idle. |
| 1F1B, GPipe | the pipeline schedule that alternates forward and backward passes, and the one that runs all forwards first. |
| Strong scaling, weak scaling | more GPUs for the same work, and more GPUs with more work. |
| Host | the CPU side of a rank. |
| Prefill, decode | the two phases of inference. |
| KV cache | the stored keys and values of past tokens. |
| MFU, HFU | model FLOPs utilization, the FLOP rate that the forward and backward passes require divided by a reference peak. Hardware FLOPs utilization also counts recomputation. |

| Symbol | Meaning |
|---|---|
| `M` | rows of one GEMM, the tokens that GEMM processes |
| `K`, `N` | input and output width of the weight in `Y = X @ W` |
| `X`, `W`, `Y` | GEMM input `[M, K]`, weight `[K, N]` and output `[M, N]` |
| `dY`, `dX`, `dW` | the gradients of the loss with respect to `Y`, `X` and `W` |
| `P` | parameters of the model |
| `T` | tokens one rank processes per step, `T = B × S` |
| `B` | rows in one rank's batch |
| `S` | tokens per row, the row length |
| `H` | hidden size; in the count for attention, the number of heads times the head width |
| `L` | decoder layers |
| `AI` | arithmetic intensity |
| `r`, `M*` | the ridge in the crossover formula, and the crossover M |
| `E`, `k` | experts per MoE layer, and experts each token is routed to |
| `Q`, `K`, `V`, `d` | queries, keys and values, and the head width, in the section on attention only |
| `R` | ranks in a group or collective |
| `D` | bytes of the tensor a collective moves |
| `p`, `m`, `i` | pipeline stages, microbatches per optimizer step, and the index of a stage counted from 0 |
| `f` | the fraction of a step that speeds up with more GPUs |

| Named quantity on a B300, bf16 | Value | Kind |
|---|---|---|
| sustained GEMM peak | about 1,400 TFLOP/s: 1,388 at 8192³, up to 1,472 | measured |
| burst GEMM peak | about 1,800 to 1,900 TFLOP/s for about the first 40 ms | measured |
| MFU reference peak | 2,250 TFLOP/s | specification |
| ceiling at the maximum clock | 2,464 TFLOP/s | computed |
| HBM bandwidth | 6.6 TB/s for a copy, read plus write, against a specified 8 TB/s | measured |
| ridge | about 210 FLOP/byte | computed |
| launch floor | about 5 µs for a trivial kernel and 8 to 12 µs for a matmul, depending on the shape and the process | measured |
| a small GEMM on the device | about 2.5 to 5 µs at small `M`, rising with `K`, plus a share that grows with the bytes of the weight | measured |
| board power limit | 1,100 W, the board's default and maximum as the driver reports them | specification |

## Further reading

Halo's own pages go deeper into the mechanisms: [Optimization](../optimization/README.md) for the kernels and [Parallelism](../parallelism/README.md) for the modes. Measured throughput of whole configurations is on [Throughput Benchmarks](../optimization/throughput-benchmarks.md).

- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). The compute-bound, memory-bound and overhead-bound regimes.
- Williams, Waterman and Patterson, [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785), Communications of the ACM, 2009. The roofline model.
- NVIDIA, [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) and [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html). Tiles and wave quantization are in the first, the tail effect in the second, arithmetic intensity in both.
- NVIDIA, the [HGX specifications](https://www.nvidia.com/en-us/data-center/hgx/), [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era), [Tuning CUDA Applications for Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide) and the CUDA Programming Guide's pages on [compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) and [lazy loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html). The limits and sizes quoted as specifications.
- Colfax Research, [CUTLASS tutorials](https://research.colfax-intl.com/category/papers/tutorials/), starting with [the design of a GEMM kernel](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/). GEMM kernel design with pipelining, and on Blackwell tensor memory, clusters and block scaling. Also their articles on [FlashAttention-3](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/) and [FlashAttention-4](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/).
- The Triton documentation, [The 5th Generation TensorCore](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html). A Blackwell matmul written step by step.
- The attention papers: FlashAttention ([arXiv 2205.14135](https://arxiv.org/abs/2205.14135)), FlashAttention-3 ([arXiv 2407.08608](https://arxiv.org/abs/2407.08608)) and FlashAttention-4 ([arXiv 2603.05451](https://arxiv.org/abs/2603.05451)).
- PyTorch's notes on [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html), [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) and [automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html), its [design note on DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html), and its tutorial on the [flight recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html) for hung jobs.
- The NCCL documentation on [environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html), and the [performance notes of nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) for the definition of bus bandwidth.
- Aleksa Gordić, [Inside TPU and GPU Clusters: The Anatomy of Collective Communication](https://www.aleksagordic.com/blog/collective-operations). Ring and tree algorithms, cost models per topology, NVLink SHARP measured.
- The papers behind the modes: ZeRO ([arXiv 1910.02054](https://arxiv.org/abs/1910.02054)), GPipe ([arXiv 1811.06965](https://arxiv.org/abs/1811.06965)), the Megatron pipeline paper ([arXiv 2104.04473](https://arxiv.org/abs/2104.04473)), sequence parallelism and selective recomputation ([arXiv 2205.05198](https://arxiv.org/abs/2205.05198)) and DeepSpeed-Ulysses ([arXiv 2309.14509](https://arxiv.org/abs/2309.14509)).
- DeepSeek-V3 ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437)) for fp8 training at scale and router balancing, and Revisiting BFloat16 Training ([arXiv 2010.06192](https://arxiv.org/abs/2010.06192)) for stochastic rounding.
- PaLM ([arXiv 2204.02311](https://arxiv.org/abs/2204.02311)), section 4.1 and appendix B, for the definition of MFU.
- Hugging Face and nanotron, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook). Data, tensor, pipeline and context parallelism with activation recomputation.
- Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering). Networking between nodes, benchmarking a cluster's links and debugging hung distributed jobs.
- [GPU MODE](https://www.youtube.com/@GPUMODE) lectures, with code in the [lectures repository](https://github.com/gpu-mode/lectures): lecture 7 on advanced quantization, lecture 12 on Flash Attention, lecture 17 on collective communication with NCCL, lecture 23 on tensor cores and lecture 36 on CUTLASS and FlashAttention-3.

## Answers

Each block answers the "Check yourself" questions that close the section it names.

### Answers on one training step

The questions are at the end of [One training step](#one-training-step).

1. Forward: 2 × 8,192 × 4,096 × 4,096 = 2.75 × 10¹¹ FLOPs. The step runs three such GEMMs: 8.25 × 10¹¹.
2. The forward GEMM and dgrad only: 5.5 × 10¹¹.

### Answers on the roofline

The questions follow [Padding and packing](#padding-and-packing), at the end of [The roofline](#the-roofline).

1. The launch and the allocation of the output. The kernel takes 0.6 to 0.8 ms.
2. 64 × 4,096 / (4,096 + 2 × 64) = 262,144 / 4,224 = 62 FLOP/byte.
3. `M*` = 209 × 2,048 / (2,048 − 418) = 263 rows. The AI can never exceed `K/2` = 1,024.
4. The weight is 2 × 6,144 × 6,144 bytes = 75.5 MB, which is 9.8 µs at 7.7 TB/s. With the 3.9 µs that is 13.7 µs. This shape was not measured. It is square and lies between two of the four fitted shapes, which missed by up to 0.5 µs, so trust it to about that. For a narrow shape the formula can be off by 1.5 µs or more.

### Answers on tiles and kernel selection

The questions follow [Rows per expert in an MoE layer](#rows-per-expert-in-an-moe-layer), at the end of [Tiles and kernel selection](#tiles-and-kernel-selection).

1. 2,050 is even and not a multiple of 8, so every GEMM with that width as `K` or `N` runs 4 to 5 times slower. Full-precision fp32 runs on the CUDA cores, which showed no such slowdown.
2. Yes, 2,880 = 360 × 8. No, 2,880 = 22.5 × 128.
3. cuBLAS picked a different kernel for the two shapes. A profiler trace gives the name of the kernel that ran, and the name carries the tile.
4. 8,192 × 4 / 128 = 256 rows, just past the crossover M. In the sweep of `M` that shape runs at about 620 TFLOP/s on the device, under half of the sustained GEMM peak.

### Answers on memory-bound kernels and fusion

The questions follow [What fusion does not help with](#what-fusion-does-not-help-with), at the end of [Memory-bound kernels and fusion](#memory-bound-kernels-and-fusion).

1. 8.75 µs, which is about the CPU's cost of launching one `torch.mm`. The kernel itself takes about 4 µs.
2. Eager, 6: the multiply reads `a` and `b` and writes an intermediate, and the add reads the intermediate and `c` and writes `x`. Fused, 4.
3. 8,192 × 200,000 = 1.64 × 10⁹ scores, which is 3.3 GB in bf16 and 6.6 GB in fp32.
4. 3 reads and 2 writes of 2 bytes are 10 bytes per parameter, 80 GB in all. At 6.65 TB/s that is 12 ms.

### Answers on attention

The questions follow [Three kernels for three generations](#three-kernels-for-three-generations), at the end of [Attention](#attention).

1. 16,384² = 2.68 × 10⁸ entries, 0.54 GB in bf16. For 32 heads in fp32: 32 × 2.68 × 10⁸ × 4 bytes = 34.4 GB, which is 32 GiB.
2. The FLOPs do not fall, and the backward pass does somewhat more. Memory falls from quadratic to linear in the row length, and traffic falls by a constant factor.
3. 128.5 / 32 is 4 copies.
4. 7 × 8,192² × 2,048 = 9.62 × 10¹¹ FLOPs. Over 0.864 ms that is 1,113 TFLOP/s, the FA4 cell of the table.

### Answers on which limit a kernel is hitting

The questions follow [The limits side by side](#the-limits-side-by-side), at the end of [Which limit a kernel is hitting](#which-limit-a-kernel-is-hitting).

1. Memory-bound: the clock is at its maximum and power is 82% of the power limit. Move fewer bytes per FLOP, with a larger `M`, fused elementwise kernels or a Flash Attention kernel.
2. Overhead-bound: power is 27% of the power limit. Look for thousands of kernels that each take a few microseconds, and for CPU work between them.
3. It is compute-bound at the clock the power cap allows. What is left is a lower precision where that helps, a kernel that does the same FLOPs for fewer watts, or more GPUs.
4. It waits for another GPU inside a communication call. Look at the other GPUs of the job, not at this one.

### Answers on precision

The questions follow [When low precision speeds training up](#when-low-precision-speeds-training-up), at the end of [Precision](#precision).

1. Between 0.5 and 1 neighboring bf16 values are 2⁻⁸ = 3.9 × 10⁻³ apart, so round-to-nearest drops the update. Stochastic rounding moves the weight one step with probability 10⁻⁴ / 3.9 × 10⁻³, about 2.6% of the time.
2. Up to 2,048. Both formats keep 10 stored mantissa bits, which is 11 significant bits, and 2¹¹ = 2,048.
3. 6 bytes per parameter is 48 GB. 12 bytes is 96 GB, and the bf16 working copy adds 16 GB.

### Answers on memory

The questions follow [Frozen weights and LoRA](#frozen-weights-and-lora), at the end of [Memory](#memory).

1. 8 bytes per parameter is 64 GB. With fp32 moments it is 12 bytes, 96 GB.
2. One hidden state is 8,192 × 2,560 × 2 bytes = 41.9 MB. Times 33 and times 36 layers that is 49.8 GB. The measured 5.87 MiB per token gives 50.4 GB.
3. A larger batch raises the `M` of every GEMM, which can lift small GEMMs above their crossover M. It also spreads the costs that are paid once per step, such as the optimizer update, over more tokens.

### Answers on collectives

The questions follow [What to do about communication](#what-to-do-about-communication), at the end of [Collectives](#collectives).

1. 2 × 7/8 × 1 GiB = 1.75 GiB, and it receives as many.
2. About the same 20 µs. Below about 1 MiB the time is the fixed cost of a call.
3. From 7/8 to 3/4 of `D`, 14% less.

### Answers on data parallelism

The questions follow [Gradient accumulation](#gradient-accumulation), at the end of [Data parallelism](#data-parallelism).

1. 2 × 7/8 × 16 GB = 28 GB.
2. Three collectives in place of two: 3 × 7/8 × 16 GB = 42 GB. In return a rank no longer holds the whole model at the peak of the step, only its shards and one or two gathered layer groups.
3. Only the optimizer state, with the reduced gradients. The gathered copy and the activations stay as they were.

### Answers on context parallelism

The questions are at the end of [Context parallelism](#context-parallelism).

1. One activation is 16,384 × 4,096 × 2 bytes = 134 MB, and the all-reduce factor is 2 × 3/4 = 1.5. That gives 4 × 36 × 1.5 × 134 MB = 29 GB.
2. 4 × 8,192 × 8 / 128 = 2,048 rows.
3. 16,384 tokens. Its activations fall to a quarter. Its weights, gradients and optimizer state stay whole.

### Answers on pipeline parallelism

The questions follow [When a pipeline is worth it](#when-a-pipeline-is-worth-it), at the end of [Pipeline parallelism](#pipeline-parallelism).

1. 3/11 = 27%. A 10% bubble needs `m = 9 × (p − 1)` = 27.
2. The bubble falls from 27% to 8.6%. Each microbatch holds a quarter of the rows it held, so every `M` falls 4 times.
3. 4 × 16,384 × 2,560 bytes = 168 MB.

### Answers on combining the modes

The questions are at the end of [Combining the modes](#combining-the-modes).

1. 16 / 2 / 2 = 4 replicas.
2. Context parallelism and a pipeline lower the activations a rank holds. FSDP at rest, tensor parallelism, expert parallelism and a pipeline divide weights. Plain data parallelism divides neither.

### Answers on how distributed runs fail

The questions are at the end of [How distributed runs fail](#how-distributed-runs-fail).

1. Amdahl: 1 / (0.1 + 0.9 / 16) = 6.4. Gustafson: 0.1 + 0.9 × 16 = 14.5.
2. Rank 3's log. It is class 6 for rank 3 and class 5 for the others, whose message blames the collective they were blocked in.
3. Every rank is waiting inside a collective, so they are waiting in different ones. Dump the stacks and read the flight recorder for the sequence number where the ranks disagree.

### Answers on training and inference

The questions follow [An RL step does both](#an-rl-step-does-both), at the end of [Training and inference](#training-and-inference).

1. About the same 8.5 to 9 µs, since 64 rows are still on the flat stretch. The rate is 64 times higher.
2. 2 × 24 × 8 × 64 × 2 bytes = 49 KB per token. 800,000 tokens hold 39 GB.
3. 234 / 53 = 4.4 s. The engine also has to load what arrives, which was more than half of one timed push.

### Answers on MFU

The questions are at the end of [MFU](#mfu).

1. 6 × 8.19 × 10⁹ × 12,000 = 590 TFLOP/s, and 590 / 2,250 is 26%.
2. No. The second forward pass of checkpointing is work that MFU does not count. HFU would count it.
3. The B300: 30% of 2,250 is 675 TFLOP/s, and 60% of 989.5 is 594.
