# Model of GPUDrainageSimulator ping-pong (gpu_drainage.rs:245-300, 346-378, 501-505)
# bind_groups[0]: read buf0 -> write buf1 ; bind_groups[1]: read buf1 -> write buf0
def simulate(steps_per_frame, frames):
    cur = 0
    version = [0, 0]   # step count whose result each buffer holds
    total = 0
    rows = []
    for f in range(frames):
        for _ in range(steps_per_frame):
            src, dst = (0, 1) if cur == 0 else (1, 0)
            total += 1
            version[dst] = total
            cur = 1 - cur
        returned = 1 - cur              # current_buffer() returns thickness_buffers[1 - current_buffer]
        latest = max(range(2), key=lambda i: version[i])
        rows.append((f, returned, version[returned], latest, version[latest]))
    return rows
for spf in (10, 9, 1):
    print(f"steps_per_frame={spf}")
    for f, ret, vret, lat, vlat in simulate(spf, 4):
        print(f"  frame {f}: current_thickness_buffer()=buf{ret} (holds step {vret}); latest=buf{lat} (step {vlat}); lag={vlat-vret} step(s)")
