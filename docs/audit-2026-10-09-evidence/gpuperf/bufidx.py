# Mirror gpu_drainage.rs:245-299 (bind_groups[0]: A->B, [1]: B->A), step() 362-377, current_buffer() 501-505
for steps in (1,2,9,10):
    cur=0; last_written=None
    for _ in range(steps):
        last_written = 1 if cur==0 else 0   # bg[0] writes B(1); bg[1] writes A(0)
        cur=1-cur
    print(f"steps_per_frame={steps:2d}: latest output buffer={last_written}, current_buffer() returns {1-cur}, caustics bound to 1 -> {'latest' if last_written==1 else 'previous sub-step'}")
print("intensity after 200 steps:", 0.999**200, " after 400:", 0.999**400)
