import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Kernel 1: Build Source Term (b)
# -----------------------------------------------------------------------------
@triton.jit
def build_b_kernel(
    u_ptr, v_ptr, b_ptr,
    rho, dt, dx, dy,
    stride_h, stride_w,
    H, W,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    pid_y = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Mask to stay within bounds. 
    # Calculation is for interior points (1:-1), so we exclude 0 and H-1.
    mask_x = pid_x < W
    mask_y = (pid_y > 0) & (pid_y < H - 1)
    mask = mask_y[:, None] & mask_x[None, :]

    # Offsets
    center_ptr = pid_y[:, None] * stride_h + pid_x[None, :] * stride_w
    
    # Neighbors with Periodic X handling
    left_x  = (pid_x[None, :] - 1 + W) % W
    right_x = (pid_x[None, :] + 1) % W
    up_y    = pid_y[:, None] + 1
    down_y  = pid_y[:, None] - 1

    # Load U
    u_r = tl.load(u_ptr + (pid_y[:, None] * stride_h + right_x * stride_w), mask=mask)
    u_l = tl.load(u_ptr + (pid_y[:, None] * stride_h + left_x * stride_w), mask=mask)
    u_u = tl.load(u_ptr + (up_y * stride_h + pid_x[None, :] * stride_w), mask=mask)
    u_d = tl.load(u_ptr + (down_y * stride_h + pid_x[None, :] * stride_w), mask=mask)

    # Load V
    v_r = tl.load(v_ptr + (pid_y[:, None] * stride_h + right_x * stride_w), mask=mask)
    v_l = tl.load(v_ptr + (pid_y[:, None] * stride_h + left_x * stride_w), mask=mask)
    v_u = tl.load(v_ptr + (up_y * stride_h + pid_x[None, :] * stride_w), mask=mask)
    v_d = tl.load(v_ptr + (down_y * stride_h + pid_x[None, :] * stride_w), mask=mask)

    # Central Differences
    dudx = (u_r - u_l) / (2.0 * dx)
    dvdy = (v_u - v_d) / (2.0 * dy)
    dudy = (u_u - u_d) / (2.0 * dy)
    dvdx = (v_r - v_l) / (2.0 * dx)

    # Source term calculation
    term1 = (1.0 / dt) * (dudx + dvdy)
    term2 = dudx * dudx
    term3 = 2.0 * (dudy * dvdx)
    term4 = dvdy * dvdy

    result = rho * (term1 - term2 - term3 - term4)
    tl.store(b_ptr + center_ptr, result, mask=mask)

# -----------------------------------------------------------------------------
# Kernel 2: Pressure Poisson Step
# -----------------------------------------------------------------------------
@triton.jit
def pressure_poisson_kernel(
    p_new_ptr, p_old_ptr, b_ptr,
    dx, dy,
    stride_h, stride_w,
    H, W,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    pid_y = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    mask_x = pid_x < W
    mask_y = pid_y < H
    mask = mask_y[:, None] & mask_x[None, :]

    # Identify Regions
    is_wall_bottom = (pid_y[:, None] == 0)
    is_wall_top    = (pid_y[:, None] == H - 1)
    is_interior    = (~is_wall_bottom) & (~is_wall_top)

    # --- Interior Logic ---
    left_x  = (pid_x[None, :] - 1 + W) % W
    right_x = (pid_x[None, :] + 1) % W
    up_y    = pid_y[:, None] + 1
    down_y  = pid_y[:, None] - 1

    # Load neighbors (from p_old)
    pn_r = tl.load(p_old_ptr + (pid_y[:, None]*stride_h + right_x*stride_w), mask=mask & is_interior)
    pn_l = tl.load(p_old_ptr + (pid_y[:, None]*stride_h + left_x*stride_w), mask=mask & is_interior)
    pn_u = tl.load(p_old_ptr + (up_y*stride_h + pid_x[None, :]*stride_w), mask=mask & is_interior)
    pn_d = tl.load(p_old_ptr + (down_y*stride_h + pid_x[None, :]*stride_w), mask=mask & is_interior)
    
    b_val = tl.load(b_ptr + (pid_y[:, None]*stride_h + pid_x[None, :]*stride_w), mask=mask & is_interior)

    # Poisson Equation
    dx2 = dx * dx
    dy2 = dy * dy
    p_computed = ((pn_r + pn_l)*dy2 + (pn_u + pn_d)*dx2) / (2*(dx2+dy2)) - (dx2*dy2)/(2*(dx2+dy2)) * b_val

    # --- Wall Logic ---
    # p[0, :] = p[1, :] (dp/dy = 0)
    # p[-1, :] = p[-2, :]
    # We read the values from p_old to keep it parallel-safe
    val_at_row1   = tl.load(p_old_ptr + (1 * stride_h + pid_x[None, :] * stride_w), mask=mask_x[None, :])
    val_at_rowHm2 = tl.load(p_old_ptr + ((H-2) * stride_h + pid_x[None, :] * stride_w), mask=mask_x[None, :])

    # Combine results
    result = tl.where(is_interior, p_computed, 0.0)
    result = tl.where(is_wall_bottom, val_at_row1, result)
    result = tl.where(is_wall_top, val_at_rowHm2, result)

    tl.store(p_new_ptr + (pid_y[:, None]*stride_h + pid_x[None, :]*stride_w), result, mask=mask)

# -----------------------------------------------------------------------------
# Kernel 3: Update Velocity
# -----------------------------------------------------------------------------
@triton.jit
def update_uv_kernel(
    u_new_ptr, v_new_ptr, u_old_ptr, v_old_ptr, p_ptr,
    rho, nu, dt, dx, dy, F,
    stride_h, stride_w,
    H, W,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    pid_y = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Interior only (1:-1), Walls (0 and H-1) remain 0
    mask_x = pid_x < W
    mask_y = (pid_y > 0) & (pid_y < H - 1)
    mask = mask_y[:, None] & mask_x[None, :]
    
    center_ptr = pid_y[:, None] * stride_h + pid_x[None, :] * stride_w

    # Indices
    left_x  = (pid_x[None, :] - 1 + W) % W
    right_x = (pid_x[None, :] + 1) % W
    up_y    = pid_y[:, None] + 1
    down_y  = pid_y[:, None] - 1

    # Load Current
    un_c = tl.load(u_old_ptr + center_ptr, mask=mask)
    vn_c = tl.load(v_old_ptr + center_ptr, mask=mask)

    # Load Neighbors
    un_r = tl.load(u_old_ptr + (pid_y[:, None]*stride_h + right_x*stride_w), mask=mask)
    un_l = tl.load(u_old_ptr + (pid_y[:, None]*stride_h + left_x*stride_w), mask=mask)
    un_u = tl.load(u_old_ptr + (up_y*stride_h + pid_x[None, :]*stride_w), mask=mask)
    un_d = tl.load(u_old_ptr + (down_y*stride_h + pid_x[None, :]*stride_w), mask=mask)

    vn_r = tl.load(v_old_ptr + (pid_y[:, None]*stride_h + right_x*stride_w), mask=mask)
    vn_l = tl.load(v_old_ptr + (pid_y[:, None]*stride_h + left_x*stride_w), mask=mask)
    vn_u = tl.load(v_old_ptr + (up_y*stride_h + pid_x[None, :]*stride_w), mask=mask)
    vn_d = tl.load(v_old_ptr + (down_y*stride_h + pid_x[None, :]*stride_w), mask=mask)

    p_r = tl.load(p_ptr + (pid_y[:, None]*stride_h + right_x*stride_w), mask=mask)
    p_l = tl.load(p_ptr + (pid_y[:, None]*stride_h + left_x*stride_w), mask=mask)
    p_u = tl.load(p_ptr + (up_y*stride_h + pid_x[None, :]*stride_w), mask=mask)
    p_d = tl.load(p_ptr + (down_y*stride_h + pid_x[None, :]*stride_w), mask=mask)

    # Updates
    u_adv = un_c * dt/dx * (un_c - un_l) + vn_c * dt/dy * (un_c - un_d)
    u_press = dt / (2*rho*dx) * (p_r - p_l)
    
    # Replaced dx**2 with dx*dx and dy**2 with dy*dy
    dx2 = dx * dx
    dy2 = dy * dy
    
    u_diff = nu * (dt/dx2 * (un_r - 2*un_c + un_l) + dt/dy2 * (un_u - 2*un_c + un_d))
    
    v_adv = un_c * dt/dx * (vn_c - vn_l) + vn_c * dt/dy * (vn_c - vn_d)
    v_press = dt / (2*rho*dy) * (p_u - p_d)
    v_diff = nu * (dt/dx2 * (vn_r - 2*vn_c + vn_l) + dt/dy2 * (vn_u - 2*vn_c + vn_d))

    tl.store(u_new_ptr + center_ptr, un_c - u_adv - u_press + u_diff + F*dt, mask=mask)
    tl.store(v_new_ptr + center_ptr, vn_c - v_adv - v_press + v_diff, mask=mask)

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def channel_flow(nit, u, v, dt, dx, dy, p, rho, nu, F):
    # u, v, p are assumed to be PyTorch tensors on the correct device (GPU)
    # We clone them to float32 if needed, though usually strict types are preferred.
    
    u_dev, v_dev, p_dev = u, v, p
    H, W = u_dev.shape
    
    # 2. Allocation
    b_dev = torch.zeros_like(u_dev)
    u_new = torch.zeros_like(u_dev)
    v_new = torch.zeros_like(v_dev)
    p_new = torch.zeros_like(p_dev)

    # 3. Kernel Configuration
    BLOCK_X, BLOCK_Y = 16, 16
    grid = (triton.cdiv(W, BLOCK_X), triton.cdiv(H, BLOCK_Y))
    
    udiff = 1.0
    stepcount = 0

    # 4. Main Simulation Loop
    while udiff > 0.001:
        # Step A: Build 'b' (Source term)
        build_b_kernel[grid](
            u_dev, v_dev, b_dev,
            rho, dt, dx, dy,
            u_dev.stride(0), u_dev.stride(1),
            H, W, BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
        )

        # Step B: Pressure Poisson (iterative)
        # We use double buffering for P
        current_p = p_dev
        next_p = p_new
        
        for _ in range(nit):
            pressure_poisson_kernel[grid](
                next_p, current_p, b_dev,
                dx, dy,
                current_p.stride(0), current_p.stride(1),
                H, W, BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
            )
            # Swap buffers
            current_p, next_p = next_p, current_p
        
        # Ensure 'p_dev' holds the latest pressure for the next step
        p_latest = current_p 

        # Step C: Update Velocity
        # Reset new buffers to 0 to ensure Walls (row 0 and H-1) are 0.
        u_new.zero_()
        v_new.zero_()

        update_uv_kernel[grid](
            u_new, v_new, u_dev, v_dev, p_latest,
            rho, nu, dt, dx, dy, F,
            u_dev.stride(0), u_dev.stride(1),
            H, W, BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
        )

        # Step D: Convergence Check
        sum_u_old = torch.sum(u_dev)
        sum_u_new = torch.sum(u_new)
        udiff = (sum_u_new - sum_u_old) / sum_u_new
        
        # Step E: Update references for next iteration
        # We copy u_new back to u_dev so the loop continues correctly
        u_dev.copy_(u_new)
        v_dev.copy_(v_new)
        p_dev.copy_(p_latest)
        
        stepcount += 1
        
    return stepcount