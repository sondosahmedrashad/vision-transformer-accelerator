`timescale 1ns/1ps

// =============================================================================
// MODULAR 64×64 MMU - PROPER K-TILING FOR 32×32 ARRAYS
// =============================================================================
// Key Architecture:
// - 32×32 systolic arrays can handle M=32, N=32, K=32 per tile
// - For 64×64 × 64×64: Need K-tiling at 32-element granularity
// - Each 64×64 operation requires 2 K-tiles (K=0-31, K=32-63)
// - All 4 arrays must process BOTH K-tiles and accumulate
// =============================================================================

module pe_8bit (
    input  wire clk,
    input  wire rst,
    input  wire clear_acc,
    input  wire valid_in,
    input  wire signed [7:0] a_in,
    input  wire signed [7:0] b_in,
    output reg  valid_out,
    output reg  signed [7:0] a_out,
    output reg  signed [7:0] b_out,
    output reg  signed [63:0] c_out
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 0;
            a_out     <= 0;
            b_out     <= 0;
            c_out     <= 0;
        end else begin
            valid_out <= valid_in;
            a_out     <= a_in;
            b_out     <= b_in;
            
            if (clear_acc)
                c_out <= 0;      // Reset accumulator
            else if (valid_in)
                c_out <= c_out + (a_in * b_in);  // Accumulate
        end
    end

endmodule

module systolic_array_32x32 (
    input  wire clk,
    input  wire rst,
    input  wire enable,
    input  wire valid_in,
    input  wire clear_acc,
    input  wire signed [7:0] matrix_a_in [0:31],    // 32 A values
    input  wire signed [7:0] matrix_b_in [0:31],    // 32 B values
    output wire valid_out,
    output wire signed [63:0] matrix_c_out [0:31][0:31] // 32×32 C results
);

    // -----------------------------------------------------------------------
    // Input skew shift-registers
    // Row i of A is delayed by i cycles so that A[i][k] and B[k][j] arrive
    // at PE[i][j] on the same clock edge (after i+j pipeline stages each).
    // Column j of B is delayed by j cycles for the same reason.
    // valid_in is also delayed by i cycles along the A-skew path so it
    // tracks the data window exactly.
    // -----------------------------------------------------------------------
    wire signed [7:0] a_skewed  [0:31];   // A after row-skew
    wire signed [7:0] b_skewed  [0:31];   // B after col-skew
    wire              v_skewed  [0:31];   // valid after row-skew

    genvar sk;
    generate
        // Row 0 / Col 0 need no delay
        assign a_skewed[0] = enable ? matrix_a_in[0] : 8'sd0;
        assign b_skewed[0] = enable ? matrix_b_in[0] : 8'sd0;
        assign v_skewed[0] = valid_in && enable;

        for (sk = 1; sk < 32; sk = sk + 1) begin : skew_gen
            // A-row skew: sk-stage shift register
            reg signed [7:0] a_sr [0:sk-1];
            reg              v_sr [0:sk-1];
            integer           idx;
            always @(posedge clk or posedge rst) begin
                if (rst) begin
                    for (idx = 0; idx < sk; idx = idx + 1) begin
                        a_sr[idx] <= 8'sd0;
                        v_sr[idx] <= 1'b0;
                    end
                end else begin
                    a_sr[0] <= enable ? matrix_a_in[sk] : 8'sd0;
                    v_sr[0] <= valid_in && enable;
                    for (idx = 1; idx < sk; idx = idx + 1) begin
                        a_sr[idx] <= a_sr[idx-1];
                        v_sr[idx] <= v_sr[idx-1];
                    end
                end
            end
            assign a_skewed[sk] = a_sr[sk-1];
            assign v_skewed[sk] = v_sr[sk-1];

            // B-col skew: sk-stage shift register
            reg signed [7:0] b_sr [0:sk-1];
            always @(posedge clk or posedge rst) begin
                if (rst) begin
                    for (idx = 0; idx < sk; idx = idx + 1)
                        b_sr[idx] <= 8'sd0;
                end else begin
                    b_sr[0] <= enable ? matrix_b_in[sk] : 8'sd0;
                    for (idx = 1; idx < sk; idx = idx + 1)
                        b_sr[idx] <= b_sr[idx-1];
                end
            end
            assign b_skewed[sk] = b_sr[sk-1];
        end
    endgenerate

    wire signed [7:0] a_wire [0:31][0:31];
    wire signed [7:0] b_wire [0:31][0:31];
    wire valid_wire [0:31][0:31];
    wire signed [63:0] c_wire [0:31][0:31];
    
    genvar i, j;
    generate
        for (i = 0; i < 32; i = i + 1) begin: row
            for (j = 0; j < 32; j = j + 1) begin: col
                pe_8bit pe_inst (
                    .clk(clk),
                    .rst(rst),
                    .clear_acc(clear_acc),
                    // valid: row-0/col-0 gets skewed valid, others propagate right through a_wire
                    .valid_in( (j==0) ? v_skewed[i] : valid_wire[i][j-1] ),
                    .a_in( (j==0) ? a_skewed[i] : a_wire[i][j-1] ),
                    .b_in( (i==0) ? b_skewed[j] : b_wire[i-1][j] ),
                    .valid_out(valid_wire[i][j]),
                    .a_out(a_wire[i][j]),
                    .b_out(b_wire[i][j]),
                    .c_out(c_wire[i][j])
                );
                assign matrix_c_out[i][j] = c_wire[i][j];
            end
        end
    endgenerate
    
    assign valid_out = enable ? valid_wire[31][31] : 1'b0;

endmodule

module mmu_modular_64x64 (
    input  wire clk,
    input  wire rst,
    
    input  wire start,
    input  wire [1:0] mode,     // Operating mode: 00=64x64, 01=64x32, 10=Dual 64x32
    input  wire [15:0] K_dim,   // K dimension size (for tiling control) can be > 64
    output wire done,
    output wire busy,
    
    // Data inputs - provide 64 elements, MMU will tile K-dimension internally
    input  wire signed [7:0] matrix_a_row [0:63],
    input  wire signed [7:0] matrix_b_col [0:63],
    input  wire signed [7:0] matrix_a_row_b [0:63],     // For dual mode
    input  wire signed [7:0] matrix_b_col_b [0:63],     // For dual mode
    
    output reg signed [63:0] result_c [0:63][0:63],
    output reg signed [63:0] result_c_b [0:63][0:63],
    output reg result_valid,
    output reg result_valid_b
);

    localparam MODE_SINGLE_64x64 = 2'd0;    // Use all 4 arrays for one 64×64 operation
    localparam MODE_SINGLE_64x32 = 2'd1;    // Use 2 arrays for one 64×32 operation (50% power)
    localparam MODE_DUAL_64x32   = 2'd2;    // Use all 4 arrays for two independent 64×32 operations simultaneously
    
    typedef enum logic [3:0] {
        IDLE,         // Waiting for start signal
        CLEAR_ACC,    // Clear accumulators (only for K-tile 0)
        PREP_FEED,    // 1-cycle buffer after clearing
        FEED,         // Feeding data for current K-tile (32 cycles: feed_cycle 0-31)
        WAIT_FLUSH,   // Wait for data to propagate through arrays
        SAVE_RESULT,  // Copy results from arrays to output
        NEXT_K_TILE,  // Check if more K-tiles needed
        DONE_STATE    // Operation complete
    } state_t;
    
    state_t state;
    
    // K-tiling at 32-element granularity for 32×32 arrays
    logic [15:0] k_tile_32;     // Current K-tile index 
    logic [15:0] k_tiles_32;    // Total number of K-tiles
    logic [15:0] feed_cycle;    
    logic [15:0] flush_cycle;    
    
    // For 64×64 operation: K=64 needs 2 tiles of 32 each
    // For 64×32 operation: K=64 needs 2 tiles of 32 each
    assign k_tiles_32 = (K_dim + 31) / 32;
    
    logic enable_00, enable_01, enable_10, enable_11;
    logic valid_in_00, valid_in_01, valid_in_10, valid_in_11;
    logic clear_acc_sig;
    logic valid_out_00, valid_out_01, valid_out_10, valid_out_11;
    
    logic signed [7:0] a_in_00 [0:31];
    logic signed [7:0] a_in_01 [0:31];
    logic signed [7:0] a_in_10 [0:31];
    logic signed [7:0] a_in_11 [0:31];
    
    logic signed [7:0] b_in_00 [0:31];
    logic signed [7:0] b_in_01 [0:31];
    logic signed [7:0] b_in_10 [0:31];
    logic signed [7:0] b_in_11 [0:31];
    
    logic signed [63:0] c_out_00 [0:31][0:31];
    logic signed [63:0] c_out_01 [0:31][0:31];
    logic signed [63:0] c_out_10 [0:31][0:31];
    logic signed [63:0] c_out_11 [0:31][0:31];
    
    systolic_array_32x32 array_00 (         
        .clk(clk), .rst(rst), .enable(enable_00),
        .valid_in(valid_in_00), .clear_acc(clear_acc_sig),
        .matrix_a_in(a_in_00), .matrix_b_in(b_in_00),
        .valid_out(valid_out_00), .matrix_c_out(c_out_00)
    );  // Top-left
    
    systolic_array_32x32 array_01 (
        .clk(clk), .rst(rst), .enable(enable_01),
        .valid_in(valid_in_01), .clear_acc(clear_acc_sig),
        .matrix_a_in(a_in_01), .matrix_b_in(b_in_01),
        .valid_out(valid_out_01), .matrix_c_out(c_out_01)
    );  // Top-right   
    
    systolic_array_32x32 array_10 (
        .clk(clk), .rst(rst), .enable(enable_10),
        .valid_in(valid_in_10), .clear_acc(clear_acc_sig),
        .matrix_a_in(a_in_10), .matrix_b_in(b_in_10),
        .valid_out(valid_out_10), .matrix_c_out(c_out_10)
    );  // Bottom-left 
    
    systolic_array_32x32 array_11 (
        .clk(clk), .rst(rst), .enable(enable_11),
        .valid_in(valid_in_11), .clear_acc(clear_acc_sig),
        .matrix_a_in(a_in_11), .matrix_b_in(b_in_11),
        .valid_out(valid_out_11), .matrix_c_out(c_out_11)
    );  // Bottom-right
    
    always_comb begin
        case (mode)
            MODE_SINGLE_64x64: begin
                enable_00 = 1'b1;
                enable_01 = 1'b1;
                enable_10 = 1'b1;
                enable_11 = 1'b1;
            end
            MODE_SINGLE_64x32: begin
                enable_00 = 1'b1;
                enable_01 = 1'b0;
                enable_10 = 1'b1;
                enable_11 = 1'b0;
            end
            MODE_DUAL_64x32: begin
                enable_00 = 1'b1;
                enable_01 = 1'b1;
                enable_10 = 1'b1;
                enable_11 = 1'b1;
            end
            default: begin
                enable_00 = 1'b0;
                enable_01 = 1'b0;
                enable_10 = 1'b0;
                enable_11 = 1'b0;
            end
        endcase
    end
    
    integer i, j;
    integer ii, jj;  // Separate loop variables for sequential block
    
    // =========================================================================
    // DATA ROUTING WITH K-TILING
    // =========================================================================
    // For K-tile 0: feed K elements 0-31  (feed_cycle 0-31)
    // For K-tile 1: feed K elements 32-63 (feed_cycle 0-31, k_offset=32)
    // Each tile takes exactly 32 feed cycles + 32 flush cycles to drain
    //
    // KEY: Each 32x32 array uses LOCAL indices 0-31
    // - Arrays 00, 01: Use A rows 0-31 (global), local indices 0-31
    // - Arrays 10, 11: Use A rows 32-63 (global), local indices 0-31
    
    always_comb begin
        for (i = 0; i < 32; i = i + 1) begin
            a_in_00[i] = 8'sd0;
            a_in_01[i] = 8'sd0;
            a_in_10[i] = 8'sd0;
            a_in_11[i] = 8'sd0;
            b_in_00[i] = 8'sd0;
            b_in_01[i] = 8'sd0;
            b_in_10[i] = 8'sd0;
            b_in_11[i] = 8'sd0;
        end
        
        if (state == FEED) begin
            case (mode)
                MODE_SINGLE_64x64: begin
                    // All arrays get data from their respective row/col slices
                    // Wavefront uses LOCAL 32x32 indices
                    for (i = 0; i < 32; i = i + 1) begin
                        // Arrays use local row index i, which maps to:
                        // Array 00, 01: global row i
                        // Array 10, 11: global row (32+i)
                        
                        a_in_00[i] = matrix_a_row[i];          // Global row 0-31
                        b_in_00[i] = matrix_b_col[i];          // Global col 0-31
                        
                        a_in_01[i] = matrix_a_row[i];          // Global row 0-31
                        b_in_01[i] = matrix_b_col[32 + i];     // Global col 32-63
                        
                        a_in_10[i] = matrix_a_row[32 + i];     // Global row 32-63
                        b_in_10[i] = matrix_b_col[i];          // Global col 0-31
                        
                        a_in_11[i] = matrix_a_row[32 + i];     // Global row 32-63
                        b_in_11[i] = matrix_b_col[32 + i];     // Global col 32-63
                    end
                end
                
                MODE_SINGLE_64x32: begin
                    for (i = 0; i < 32; i = i + 1) begin
                        a_in_00[i] = matrix_a_row[i];
                        b_in_00[i] = matrix_b_col[i];
                        a_in_10[i] = matrix_a_row[32 + i];
                        b_in_10[i] = matrix_b_col[i];
                    end
                end
                
                MODE_DUAL_64x32: begin
                    for (i = 0; i < 32; i = i + 1) begin
                        a_in_00[i] = matrix_a_row[i];
                        b_in_00[i] = matrix_b_col[i];
                        a_in_10[i] = matrix_a_row[32 + i];
                        b_in_10[i] = matrix_b_col[i];
                        
                        a_in_01[i] = matrix_a_row_b[i];
                        b_in_01[i] = matrix_b_col_b[i];
                        a_in_11[i] = matrix_a_row_b[32 + i];
                        b_in_11[i] = matrix_b_col_b[i];
                    end
                end
            endcase
        end
    end
    
    always_comb begin
        // valid_in is high for the first 32 cycles of FEED (actual K data).
        // FEED runs for 63 cycles total to flush the 31-deep A-skew registers.
        if (state == FEED && feed_cycle < 32) begin
            valid_in_00 = enable_00;
            valid_in_01 = enable_01;
            valid_in_10 = enable_10;
            valid_in_11 = enable_11;
        end else begin
            valid_in_00 = 1'b0;
            valid_in_01 = 1'b0;
            valid_in_10 = 1'b0;
            valid_in_11 = 1'b0;
        end
    end
    
    assign done = (state == DONE_STATE);
    assign busy = (state != IDLE);
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            k_tile_32 <= 0;
            feed_cycle <= 0;
            flush_cycle <= 0;
            clear_acc_sig <= 0;
            result_valid <= 0;
            result_valid_b <= 0;
            
            for (ii = 0; ii < 64; ii = ii + 1) begin
                for (jj = 0; jj < 64; jj = jj + 1) begin
                    result_c[ii][jj] <= 0;
                    result_c_b[ii][jj] <= 0;
                end
            end
            
        end else begin
            clear_acc_sig <= 0;
            result_valid <= 0;
            result_valid_b <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= CLEAR_ACC;
                        k_tile_32 <= 0;
                        feed_cycle <= 0;
                        flush_cycle <= 0;
                    end
                end
                
                CLEAR_ACC: begin
                    // Only clear on first K-tile
                    clear_acc_sig <= (k_tile_32 == 0) ? 1'b1 : 1'b0;
                    state <= PREP_FEED;  // Go to prep state first
                    feed_cycle <= 0;
                end
                
                PREP_FEED: begin
                    // One cycle delay to ensure clear completes before feeding
                    clear_acc_sig <= 0;
                    state <= FEED;
                    feed_cycle <= 0;
                end
                
                FEED: begin
                    // Feed for 63 cycles per K-tile:
                    //   Cycles 0-31: real K data (valid_in is high externally)
                    //   Cycles 32-62: A-skew registers drain (valid_in stays low externally)
                    // This ensures even row-31's skew register has flushed before WAIT_FLUSH.
                    if (feed_cycle < 62) begin
                        feed_cycle <= feed_cycle + 1;
                    end else begin
                        state <= WAIT_FLUSH;
                        flush_cycle <= 0;
                    end
                end
                
                WAIT_FLUSH: begin
                    // After 63-cycle FEED, last data entered PE[31][0] at feed_cycle=62.
                    // It propagates right to PE[31][31] in 31 more A-pipeline cycles.
                    // Valid also propagates rightward 31 more steps. 32 cycles is safe.
                    if (flush_cycle < 32) begin
                        flush_cycle <= flush_cycle + 1;
                    end else begin
                        state <= NEXT_K_TILE;
                    end
                end
                
                NEXT_K_TILE: begin
                    if (k_tile_32 >= k_tiles_32 - 1) begin
                        // Finished all K-tiles, save result
                        state <= SAVE_RESULT;
                    end else begin
                        // More K-tiles to process
                        k_tile_32 <= k_tile_32 + 1;
                        state <= CLEAR_ACC;
                    end
                end
                
                SAVE_RESULT: begin
                    case (mode)
                        MODE_SINGLE_64x64: begin
                            for (ii = 0; ii < 32; ii = ii + 1) begin
                                for (jj = 0; jj < 32; jj = jj + 1) begin
                                    result_c[ii][jj] <= c_out_00[ii][jj];
                                    result_c[ii][32+jj] <= c_out_01[ii][jj];
                                    result_c[32+ii][jj] <= c_out_10[ii][jj];
                                    result_c[32+ii][32+jj] <= c_out_11[ii][jj];
                                end
                            end
                            result_valid <= 1;
                        end
                        
                        MODE_SINGLE_64x32: begin
                            for (ii = 0; ii < 32; ii = ii + 1) begin
                                for (jj = 0; jj < 32; jj = jj + 1) begin
                                    result_c[ii][jj] <= c_out_00[ii][jj];
                                    result_c[32+ii][jj] <= c_out_10[ii][jj];
                                end
                            end
                            result_valid <= 1;
                        end
                        
                        MODE_DUAL_64x32: begin
                            for (ii = 0; ii < 32; ii = ii + 1) begin
                                for (jj = 0; jj < 32; jj = jj + 1) begin
                                    result_c[ii][jj] <= c_out_00[ii][jj];
                                    result_c[32+ii][jj] <= c_out_10[ii][jj];
                                    result_c_b[ii][jj] <= c_out_01[ii][jj];
                                    result_c_b[32+ii][jj] <= c_out_11[ii][jj];
                                end
                            end
                            result_valid <= 1;
                            result_valid_b <= 1;
                        end
                    endcase
                    
                    state <= DONE_STATE;
                end
                
                DONE_STATE: begin
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule