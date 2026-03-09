# tilelang-demo-t001

## 优化同步

### 优化前 VS 优化后
[优化前: fyd_v2_sync.orig.cpp](./test_elementwise_add_pipeline_fyd_v2_sync.orig.cpp)

[优化前: fyd_v2_sync.target.cpp](./test_elementwise_add_pipeline_fyd_v2_sync.target.cpp)

### AI Ask
```
@test_elementwise_add_pipeline_fyd_v2_sync.target.cpp @test_elementwise_add_pipeline_fyd_v2_sync.orig.cpp  请分析一下这两个代码的实现逻辑，哪个实现的性能更好，为什么？
```

### 核心逻辑对比
两个实现都是昇腾AICore上的逐元素加法流水线实现，采用双缓冲(double buffering)技术重叠数据传输和计算：
1. **orig版本**：使用固定事件值和PipeBarrier全局屏障
2. **target版本**：使用奇偶ping-pong事件标志，移除冗余全局屏障

### 性能结论：**target版本性能更好**

#### 关键优化点：
1. **更高效的同步机制**
   - orig使用固定事件值(1/5/2/3等)和全局`PipeBarrier`，同步粒度粗，容易造成不必要的等待
   - target使用`n_i % 2`作为事件标志，实现双缓冲的精确同步，同步粒度更细，减少等待时间

2. **消除冗余屏障**
   - orig在循环中有两处`PipeBarrier<PIPE_MTE2/PIPE_MTE3>`全局屏障，会阻塞整个流水线
   - target完全移除了全局屏障，仅使用事件标志实现模块间同步，流水线阻塞更少

3. **更合理的事件配对**
   - orig事件配对混乱(MTE2_V=1, MTE3_V=5等)，容易导致同步错误或过度等待
   - target事件配对清晰：MTE2_V/MTE3_V和V_MTE2/V_MTE3成对使用`n_i % 2`，同步逻辑更高效

4. **数据访问模式相同**
   - 两者分块策略、内存访问粒度(32x64)、计算量(2048元素/次)完全一致，没有额外访存开销

target版本通过更精细的同步机制和消除冗余屏障，实现了更高的流水线利用率，性能优于orig版本。