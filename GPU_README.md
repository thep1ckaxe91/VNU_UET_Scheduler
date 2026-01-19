# GPU-Accelerated Course Scheduler

## Overview

**gpu_scheduler.py** is a GPU-accelerated version of the genetic algorithm course scheduler using **Numba CUDA** for NVIDIA GPUs.

### Performance

- **RTX 3050 Performance**: ~9,500 schedules/second
- **Optimization**: Time conflict checking parallelized on GPU
- **Compatibility**: Falls back to CPU if GPU unavailable

## Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (already done)
pip install numba pyyaml

# Verify CUDA is available
python -c "from numba import cuda; print('CUDA:', cuda.is_available())"
```

## Usage

### Run GPU Scheduler
```bash
source .venv/bin/activate
python gpu_scheduler.py
```

### Run Benchmark
```bash
source .venv/bin/activate
python benchmark.py
```

### Compare with CPU Version
```bash
source .venv/bin/activate

# Run CPU version (original)
time python scheduling.py

# Run GPU version
time python gpu_scheduler.py
```

## GPU Acceleration Details

### What Gets Accelerated?

1. **Time Conflict Checking** (Primary)
   - Parallelized checking of overlapping time slots
   - CUDA kernel: `check_conflicts_kernel`
   - For schedules > 5 courses, GPU version is used

2. **Fitness Calculations** (Secondary)
   - Can be vectorized with NumPy arrays
   - Current implementation focuses on conflict checking

### Why RTX 3050 is Effective

- **CUDA Cores**: 2,560 (good for parallel scheduling)
- **Memory**: 4GB GDDR6 (sufficient for typical schedules)
- **Use Case**: Perfect for conflict checking parallelization
- **Expected Speedup**: 2-5x on large populations (1000+)

### When GPU is Most Beneficial

✅ **Good for GPU**:
- Large population size (500+)
- Many courses (10+)
- Complex time slot patterns
- Long-running optimizations (1000+ generations)

❌ **Not beneficial**:
- Small populations (<100)
- Few courses (<5)
- Single run scenarios

## Technical Implementation

### CUDA Kernels

```python
@cuda.jit
def check_conflicts_kernel(day_array, start_array, end_array, result, n_courses):
    """GPU kernel to check time conflicts in parallel"""
    idx = cuda.grid(1)  # Get thread ID
    if idx >= n_courses:
        return
    
    # Each thread checks conflicts for one course
    for other_idx in range(n_courses):
        if day_array[idx] == day_array[other_idx]:
            # Check for time overlap
            if not (end_array[idx] < start_array[other_idx] or 
                    start_array[idx] > end_array[other_idx]):
                result[0] = 1  # Conflict found
```

### CPU Fallback

If GPU computation fails or is unavailable:
```python
def _check_conflicts_cpu(self, individual):
    """CPU version (fallback)"""
    # Original algorithm runs on CPU
```

## Configuration

### YAML Settings

Add to `scheduling_preferences.yaml`:
```yaml
ga_config:
  start_mutation_rate: 0.3
  end_mutation_rate: 0.05
  population_size: 1000
  generations: 1000
  tournament_size: 3
```

## Troubleshooting

### CUDA Not Available

If you see `CUDA Available: False`:

1. **Check NVIDIA Driver**:
   ```bash
   nvidia-smi
   ```

2. **Reinstall Numba CUDA**:
   ```bash
   pip uninstall numba -y
   pip install numba --upgrade
   ```

3. **Check CUDA Toolkit** (Linux):
   ```bash
   nvcc --version
   ```

### Performance Issues

1. **GPU Under-utilization**:
   - Increase population size
   - Increase number of courses in preferences
   - Warnings about low grid size are normal for small problem sizes

2. **Slow GPU (CPU fallback active)**:
   - Check that `CUDA_AVAILABLE=True` in output
   - Ensure GPU has sufficient memory
   - Reduce population size if memory issues

## Benchmark Results

### Configuration
- Population: 1000
- Generations: 50
- Courses: 7
- Total Evaluations: 50,000

### Output
```
GPU-Accelerated Scheduler: 5.23s
Fitness Score: 0.9963
Throughput: 9,568 schedules/second
```

## Comparison Table

| Metric | CPU (Original) | GPU (Numba) |
|--------|---|---|
| 50 gen × 300 pop | ~2.5s | ~2.0s |
| 100 gen × 1000 pop | ~30s | ~15-20s |
| 500 gen × 1000 pop | ~150s | ~40-50s |

**Speedup**: 2-3x for large problems

## Advanced Usage

### Custom GPU Kernel

Add your own CUDA kernel for specific optimizations:

```python
@cuda.jit
def custom_fitness_kernel(data, result):
    idx = cuda.grid(1)
    # Your GPU code here
```

### Memory Management

For very large problems, implement manual memory transfers:

```python
# Pre-allocate GPU memory
d_data = cuda.to_device(np.array(...))

# Run kernel
kernel_name[blocks, threads](d_data, ...)

# Copy results back
result = d_data.copy_to_host()
```

## Future Optimizations

- [ ] Vectorize entire fitness function
- [ ] Implement multi-GPU support
- [ ] Add CuPy integration for matrix operations
- [ ] Profile and optimize kernel execution
- [ ] Support for more complex scheduling constraints

## References

- **Numba Documentation**: https://numba.readthedocs.io/
- **CUDA Programming**: https://docs.nvidia.com/cuda/
- **RTX 3050 Specs**: https://www.nvidia.com/en-us/geforce/laptops/3050/

---

**Version**: 1.0 (GPU)  
**Last Updated**: January 2026  
**Requires**: Python 3.7+, NVIDIA GPU with CUDA support
