// Fixed High-Performance SV ViT (vit_top_integrated_fixed.sv)
// Critical Fixes:
// 1. Patch embedding bias scaling corrected
// 2. Intermediate value precision maintained
// 3. Proper scaling chain through all stages
`timescale 1ns / 1ps

import dyadic_params::*;

module vit_top_integrated #(
    // Path to folder containing all .mem weight files.
    // Questa simulation : relative to the directory you run vsim from.
    // Vivado synthesis  : relative to the Vivado project directory.
    // Use forward slashes. Trailing slash required. Example: "weights/"
    parameter string WEIGHT_DIR = "weights/",
    parameter string VERSION_TAG = "V_ATTN_FIX_V11_QKV_PROJ_MLP1_MLP2_MIRROR"
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [7:0] image_in [0:2][0:63][0:63],
    output logic done,
    output logic [31:0] prediction
);

    // --- PARAMETERS ---
    localparam NUM_PATCHES = 64;
    localparam EMBED_DIM = 256;
    localparam DEPTH = 6;
    localparam NUM_HEADS = 8;
    localparam HEAD_DIM = 32;
    localparam MLP_DIM = 768;
    localparam NUM_CLASSES = 10;

    // --- WEIGHT/BIAS STORAGE ---
    logic signed [7:0]  pos_embed [NUM_PATCHES][EMBED_DIM];
    logic signed [7:0]  patch_w   [EMBED_DIM][3][8][8];
    logic signed [31:0] patch_b   [EMBED_DIM];
    
    // Transformer block weights
    logic signed [7:0]  b_norm1_w [DEPTH][EMBED_DIM]; logic signed [31:0] b_norm1_b [DEPTH][EMBED_DIM];
    logic signed [7:0]  b_qkv_w   [DEPTH][768][EMBED_DIM]; logic signed [31:0] b_qkv_b   [DEPTH][768];
    logic signed [7:0]  b_atn_p_w [DEPTH][EMBED_DIM][EMBED_DIM]; logic signed [31:0] b_atn_p_b [DEPTH][EMBED_DIM];
    logic signed [7:0]  b_norm2_w [DEPTH][EMBED_DIM]; logic signed [31:0] b_norm2_b [DEPTH][EMBED_DIM];
    logic signed [7:0]  b_mlp1_w  [DEPTH][MLP_DIM][EMBED_DIM]; logic signed [31:0] b_mlp1_b  [DEPTH][MLP_DIM];
    logic signed [7:0]  b_mlp2_w  [DEPTH][EMBED_DIM][MLP_DIM]; logic signed [31:0] b_mlp2_b  [DEPTH][EMBED_DIM];
    
    logic signed [7:0]  norm_f_w  [EMBED_DIM]; logic signed [31:0] norm_f_b  [EMBED_DIM];
    logic signed [7:0]  head_w    [10][EMBED_DIM]; logic signed [31:0] head_b    [10];

    // --- TENSORS ---
    logic signed [63:0] x [NUM_PATCHES][EMBED_DIM];
    logic signed [63:0] patch_acc [EMBED_DIM];
    
    // --- STATE MACHINE ---
    typedef enum logic [4:0] {
        ST_IDLE, ST_PATCH_EMBED, ST_POS_EMBED, ST_POS_EMBED_WAIT,
        ST_BLOCK_START, ST_LN1, ST_QKV, ST_SOFT_SCORES, ST_SOFTMAX, ST_ATTN_V, ST_PROJ, ST_RES1,
        ST_LN2, ST_MLP1, ST_GELU, ST_MLP2, ST_RES2, ST_BLOCK_NEXT,
        ST_NORM_FINAL, ST_GAP, ST_HEAD, ST_DONE
    } state_t;
    state_t state;
    logic [3:0] b_curr;
    logic [6:0] patch_cnt;
    logic [6:0] patch_cnt_j;
    logic [7:0] pixel_cnt;
    logic [3:0] head_cnt;
    logic [3:0] col_tile;   // Output column tile index (each tile = 64 output cols)

    // --- PIPELINED COMPONENT INTERFACES ---

    // Feed counters: one per MMU instance, count k-index during FEED state
    logic [8:0] mmu1_feed_k;
    logic [8:0] mmu1_p_feed_k;
    logic [8:0] mmu2_feed_k;
    logic [8:0] mmu2_p_feed_k;

    // MMU 1: used for QKV (64x256→768) and reused for other 64-col-tile ops
    logic mmu_start, mmu_done;
    logic [15:0] tile_dim;
    logic signed [7:0]  mmu1_a   [64][256];
    logic signed [63:0] mmu_out [0:63][0:63];   // MMU native 64×64 output
    logic signed [63:0] mmu1_buf [64][768];       // Legacy wide accumulation buffer for QKV (kept for compatibility/debug)

    // Generic ping-pong matrix buffer integrated for QKV only
    logic qkv_buf_wr_en;
    logic qkv_buf_swap;
    logic signed [63:0] qkv_buf_rd_matrix [0:63][0:767];

    logic signed [7:0]  mmu_row [0:63];
    logic signed [7:0]  mmu_col [0:63];
    logic signed [7:0]  mmu1_row [0:63];
    logic signed [7:0]  mmu1_col [0:63];
    logic signed [7:0]  mmu1_dummy_a [0:63];      // Unused dual-port inputs
    logic signed [7:0]  mmu1_dummy_b [0:63];
    mmu_modular_64x64 u_mmu1 (
        .clk(clk), .rst(rst), .start(mmu_start),
        .mode(2'd0), .K_dim(tile_dim),
        .done(mmu_done), .busy(),
        .matrix_a_row(mmu_row), .matrix_b_col(mmu_col),
        .matrix_a_row_b(mmu1_dummy_a), .matrix_b_col_b(mmu1_dummy_b),
        .result_c(mmu_out), .result_c_b(),
        .result_valid(), .result_valid_b()
    );
    assign mmu1_dummy_a = '{default: 8'sd0};
    assign mmu1_dummy_b = '{default: 8'sd0};

    // MMU for Attention Projection (64x256x256)
    //logic mmu_start, mmu_done;
    logic signed [7:0]  attn_out      [64][256];  // Output of attention @ V, input to proj
    //logic signed [63:0] mmu_out    [0:63][0:63]; // MMU native 64×64 output
    logic signed [63:0] mmu1_p_buf    [64][256];    // Wide accumulation buffer for proj
    logic signed [7:0]  mmu1_p_row    [0:63];
    logic signed [7:0]  mmu1_p_col    [0:63];
    logic signed [7:0]  mmu1_p_dummy_a [0:63];
    logic signed [7:0]  mmu1_p_dummy_b [0:63];

    assign mmu1_p_dummy_a = '{default: 8'sd0};
    assign mmu1_p_dummy_b = '{default: 8'sd0};

    // MMU for MLP1 (64x256x768)
    //logic mmu_start, mmu_done;
    logic signed [7:0]  mmu2_a   [64][256];       // Input to MLP1
    logic signed [63:0] mmu2_out [0:63][0:63];    // MMU native 64×64 output
    logic signed [63:0] mmu2_buf [64][768];        // Wide accumulation buffer for MLP1
    logic signed [7:0]  mmu2_row [0:63];
    logic signed [7:0]  mmu2_col [0:63];
    logic signed [7:0]  mmu2_dummy_a [0:63];
    logic signed [7:0]  mmu2_dummy_b [0:63];

    assign mmu2_dummy_a = '{default: 8'sd0};
    assign mmu2_dummy_b = '{default: 8'sd0};

    // Generic ping-pong matrix buffer mirrored for MLP1 only
    logic mlp1_buf_wr_en;
    logic mlp1_buf_swap;
    logic signed [63:0] mlp1_buf_rd_matrix [0:63][0:767];

    // MMU for MLP2 (64x768x256) — K=768, output cols=256 (4 tiles of 64)
    //logic mmu_start, mmu_done;
    logic signed [7:0]  mmu2_p_a   [64][768];      // Input to MLP2 (GELU output)
    logic signed [63:0] mmu2_p_out [0:63][0:63];   // MMU native 64×64 output
    logic signed [63:0] mmu2_p_buf [64][256];       // Wide accumulation buffer for MLP2
    logic signed [7:0]  mmu2_p_row [0:63];
    logic signed [7:0]  mmu2_p_col [0:63];
    logic signed [7:0]  mmu2_p_dummy_a [0:63];
    logic signed [7:0]  mmu2_p_dummy_b [0:63];

    assign mmu2_p_dummy_a = '{default: 8'sd0};
    assign mmu2_p_dummy_b = '{default: 8'sd0};

    // Generic ping-pong matrix buffer mirrored for MLP2 only
    logic mlp2_buf_wr_en;
    logic mlp2_buf_swap;
    logic signed [63:0] mlp2_buf_rd_matrix [0:63][0:255];

    always_comb begin
        for (int i = 0; i < 64; i++) mmu_row[i] = 8'sd0;
        for (int j = 0; j < 64; j++) mmu_col[j] = 8'sd0;
        if (u_mmu1.state == u_mmu1.FEED && u_mmu1.feed_cycle < 32) begin
            case (state)
                ST_QKV: begin
                    for (int i = 0; i < 64; i++)
                        mmu_row[i] = mmu1_a[i][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                    for (int j = 0; j < 64; j++)
                        mmu_col[j] = b_qkv_w[b_curr][col_tile * 64 + j][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                end
                ST_PROJ: begin
                    for (int i = 0; i < 64; i++)
                        mmu_row[i] = attn_out[i][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                    for (int j = 0; j < 64; j++)
                        mmu_col[j] = b_atn_p_w[b_curr][col_tile * 64 + j][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                end
                ST_MLP1: begin
                    for (int i = 0; i < 64; i++)
                        mmu_row[i] = mmu2_a[i][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                    for (int j = 0; j < 64; j++)
                        mmu_col[j] = b_mlp1_w[b_curr][col_tile * 64 + j][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                end
                ST_MLP2: begin
                    for (int i = 0; i < 64; i++)
                        mmu_row[i] = mmu2_p_a[i][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                    for (int j = 0; j < 64; j++)
                        mmu_col[j] = b_mlp2_w[b_curr][col_tile * 64 + j][u_mmu1.k_tile_32 * 32 + mmu1_feed_k];
                end
                default: begin end
            endcase
        end
    end

    // Single feed counter — all four stages use the same MMU so one counter suffices
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            mmu1_feed_k   <= 0;
            mmu1_p_feed_k <= 0;
            mmu2_feed_k   <= 0;
            mmu2_p_feed_k <= 0;
        end else begin
            mmu1_feed_k   <= (u_mmu1.state == u_mmu1.FEED) ? (mmu1_feed_k < 31 ? mmu1_feed_k + 1 : 31) : 0;
            mmu1_p_feed_k <= mmu1_feed_k;
            mmu2_feed_k   <= mmu1_feed_k;
            mmu2_p_feed_k <= mmu1_feed_k;
        end
    end

    // LayerNorm
    logic ln_start, ln_done;
    logic signed [63:0] ln_in [EMBED_DIM];
    logic signed [7:0]  ln_w [EMBED_DIM];
    logic signed [31:0] ln_b [EMBED_DIM];
    logic signed [31:0] ln_out [EMBED_DIM];
    logic signed [31:0] ln_m_fixed, ln_s_fixed;
    layernorm_pipelined #(256) u_ln (.clk(clk), .rst(rst), .start(ln_start), .x_in(ln_in), .weight(ln_w), .bias(ln_b), .m_fixed(ln_m_fixed), .s_fixed(ln_s_fixed), .x_out(ln_out), .done(ln_done));

    // GELU
    logic gelu_start, gelu_done;
    logic signed [63:0] gelu_in [64][768];
    logic signed [31:0] gelu_out [64][768];
    logic signed [31:0] gelu_m_idx, gelu_s_idx;
    gelu_pipelined #(64, 768) u_gelu (.clk(clk), .rst(rst), .start(gelu_start), .x_in(gelu_in), .m_idx(gelu_m_idx), .s_idx(gelu_s_idx), .x_out(gelu_out), .done(gelu_done));

    // Softmax (Row-based)
    logic soft_start, soft_done;
    logic signed [63:0] soft_in [64];
    logic signed [7:0] soft_out [64];
    logic signed [31:0] soft_m_idx, soft_s_idx;
    softmax_pipelined #(64) u_soft (.clk(clk), .rst(rst), .start(soft_start), .x_in(soft_in), .m_idx(soft_m_idx), .s_idx(soft_s_idx), .x_out(soft_out), .done(soft_done));

    // Weight loading
    initial begin
        $display("[%0t] Loading weights from %s...", $time, WEIGHT_DIR);
        // Primary weight/bias loading - Use 1D buffer for 4D/3D arrays to ensure simulator stability
        begin
            static logic [31:0] tmp_v_large [196608];
            static logic [31:0] tmp_b_large [768];
            
            // Pos Embed
            $readmemh({WEIGHT_DIR, "pos_embed.mem"}, tmp_v_large);
            for (int i=0; i<64; i++) for (int j=0; j<256; j++) pos_embed[i][j] = tmp_v_large[i*256 + j][7:0];

            // Patch Proj (4D)
            $readmemh({WEIGHT_DIR, "patch_embed_proj_weight.mem"}, tmp_v_large);
            for (int d=0; d<256; d++) for (int c=0; c<3; c++) for (int r=0; r<8; r++) for (int col=0; col<8; col++)
                patch_w[d][c][r][col] = tmp_v_large[d*3*8*8 + c*8*8 + r*8 + col][7:0];
            
            $readmemh({WEIGHT_DIR, "patch_embed_proj_bias.mem"}, tmp_b_large);
            for (int d=0; d<256; d++) patch_b[d] = tmp_b_large[d];
        
            // Block weights
            for (int i=0; i<6; i++) begin
                string b_idx; b_idx.itoa(i);
                
                // Norm 1
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_norm1_weight.mem"}, tmp_v_large);
                for (int j=0; j<256; j++) b_norm1_w[i][j] = tmp_v_large[j][7:0];
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_norm1_bias.mem"}, tmp_b_large);
                for (int j=0; j<256; j++) b_norm1_b[i][j] = tmp_b_large[j];
                
                // QKV
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_attn_qkv_weight.mem"}, tmp_v_large);
                for (int m=0; m<768; m++) for (int d=0; d<256; d++) b_qkv_w[i][m][d] = tmp_v_large[m*256 + d][7:0];
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_attn_qkv_bias.mem"}, tmp_b_large);
                for (int m=0; m<768; m++) b_qkv_b[i][m] = tmp_b_large[m];
                
                // Attn Proj
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_attn_proj_weight.mem"}, tmp_v_large);
                for (int m=0; m<256; m++) for (int d=0; d<256; d++) b_atn_p_w[i][m][d] = tmp_v_large[m*256 + d][7:0];
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_attn_proj_bias.mem"}, tmp_b_large);
                for (int j=0; j<256; j++) b_atn_p_b[i][j] = tmp_b_large[j];
                
                // Norm 2
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_norm2_weight.mem"}, tmp_v_large);
                for (int j=0; j<256; j++) b_norm2_w[i][j] = tmp_v_large[j][7:0];
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_norm2_bias.mem"}, tmp_b_large);
                for (int j=0; j<256; j++) b_norm2_b[i][j] = tmp_b_large[j];
                
                // MLP
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_mlp_fc1_weight.mem"}, tmp_v_large);
                for (int m=0; m<768; m++) for (int d=0; d<256; d++) b_mlp1_w[i][m][d] = tmp_v_large[m*256 + d][7:0];
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_mlp_fc1_bias.mem"}, tmp_b_large);
                for (int m=0; m<768; m++) b_mlp1_b[i][m] = tmp_b_large[m];
                
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_mlp_fc2_weight.mem"}, tmp_v_large);
                for (int m=0; m<256; m++) for (int d=0; d<768; d++) b_mlp2_w[i][m][d] = tmp_v_large[m*768 + d][7:0];
                $readmemh({WEIGHT_DIR, "blocks_", b_idx, "_mlp_fc2_bias.mem"}, tmp_b_large);
                for (int j=0; j<256; j++) b_mlp2_b[i][j] = tmp_b_large[j];
            end
            
            // Final Norm
            $readmemh({WEIGHT_DIR, "norm_weight.mem"}, tmp_v_large);
            for (int j=0; j<256; j++) norm_f_w[j] = tmp_v_large[j][7:0];
            $readmemh({WEIGHT_DIR, "norm_bias.mem"}, tmp_b_large);
            for (int j=0; j<256; j++) norm_f_b[j] = tmp_b_large[j];
            
            // Head
            $readmemh({WEIGHT_DIR, "head_weight.mem"}, tmp_v_large);
            for (int c=0; c<10; c++) for (int j=0; j<256; j++) head_w[c][j] = tmp_v_large[c*256 + j][7:0];
            $readmemh({WEIGHT_DIR, "head_bias.mem"}, tmp_b_large);
            for (int c=0; c<10; c++) head_b[c] = tmp_b_large[c];
        end
        
        if (patch_w[0][0][0][0] === 8'hx) $display("CRITICAL ERROR: Weights failed to load!");
        else begin
            $display("SUCCESS: Weights loaded.");
            $display("  patch_w[0][0][0][0]  = %d (Expected: -21)", $signed(patch_w[0][0][0][0]));
            $display("  b_norm1_w[0][0]      = %d (Expected: 60)", $signed(b_norm1_w[0][0]));
            $display("  b_norm1_b[0][0]      = %d (Expected: 1039113344)", $signed(b_norm1_b[0][0]));
            $display("  b_norm1_w[5][0]      = %d (Expected: 85)", $signed(b_norm1_w[5][0]));
            $display("  norm_f_w[0]          = %d (Expected: 45)", $signed(norm_f_w[0]));
        end
    end

    // Storage for Q, K, V (High Precision for Attention)
    logic signed [31:0] q [NUM_HEADS][64][32];
    logic signed [31:0] k [NUM_HEADS][64][32];
    logic signed [31:0] v [NUM_HEADS][64][32];
    logic signed [63:0] row_scores [64];
    logic signed [7:0]  attn_weights [NUM_HEADS][64][64];
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= ST_IDLE; done <= 0; b_curr <= 0;
            ln_start <= 0; mmu_start <= 0; //mmu_start <= 0; mmu_start <= 0; mmu_start <= 0;
            gelu_start <= 0; soft_start <= 0;
            qkv_buf_wr_en <= 0; qkv_buf_swap <= 0;
            patch_cnt <= 0; pixel_cnt <= 0; col_tile <= 0;
            for (int i=0; i<64; i++) for (int d=0; d<256; d++) attn_out[i][d] <= 8'sd0;
            for (int i=0; i<64; i++) for (int d=0; d<768; d++) begin mmu1_buf[i][d] <= 0; mmu2_buf[i][d] <= 0; end
            for (int i=0; i<64; i++) for (int d=0; d<256; d++) begin mmu1_p_buf[i][d] <= 0; mmu2_p_buf[i][d] <= 0; end
            for (int h=0; h<8; h++) for (int i=0; i<64; i++) for (int j=0; j<64; j++) attn_weights[h][i][j] <= 0;
            for (int h=0; h<8; h++) for (int i=0; i<64; i++) for (int d=0; d<32; d++) begin
                q[h][i][d] <= 0; k[h][i][d] <= 0; v[h][i][d] <= 0;
            end
        end else begin
            // one-cycle pulses for QKV buffer control
            qkv_buf_wr_en <= 0;
            qkv_buf_swap  <= 0;
            case (state)
                ST_IDLE: if (start) begin 
                    state <= ST_PATCH_EMBED; patch_cnt <= 0; pixel_cnt <= 0; b_curr <= 0;
                    $display("[%0t] === STARTING INFERENCE - VERSION: %s ===", $time, VERSION_TAG);
                    $display("[%0t] First 4 pixels of image[0][0][0:3]: %d %d %d %d", $time, 
                             $signed(image_in[0][0][0]), $signed(image_in[0][0][1]), 
                             $signed(image_in[0][0][2]), $signed(image_in[0][0][3]));
                end
                
                ST_PATCH_EMBED: begin
                    if (patch_cnt < 64) begin
                        automatic int chan = pixel_cnt / 64; 
                        automatic int row = (pixel_cnt % 64) / 8; // Row within 8x8 patch
                        automatic int col = pixel_cnt % 8;        // Col within 8x8 patch
                        automatic int img_r = (patch_cnt / 8) * 8 + row; 
                        automatic int img_c = (patch_cnt % 8) * 8 + col;
                        
                        if (pixel_cnt < 192) begin
                            // Use local automatic accumulation for precision and safety
                            for(int d=0; d<256; d++) begin
                                automatic logic signed [127:0] b_val = $signed(patch_b[d]);
                                automatic logic signed [127:0] b_mul = $signed(dyadic_params::patch_embed_bias_m);
                                automatic logic signed [127:0] b_scaled = (b_val * b_mul) >>> dyadic_params::patch_embed_bias_s;
                                
                                automatic logic signed [63:0] s_pix = 64'($signed(image_in[chan][img_r][img_c]));
                                automatic logic signed [63:0] s_wei = $signed(patch_w[d][chan][row][col]);
                                
                                // On pixel 0, start with bias + first product
                                if (pixel_cnt == 0) patch_acc[d] <= b_scaled[63:0] + (s_pix * s_wei);
                                else patch_acc[d] <= patch_acc[d] + (s_pix * s_wei);
                            end
                            if (patch_cnt == 0 && pixel_cnt == 0) begin
                                $display("[%0t]   [PATCH_EMBED Trace] P0 D0: pix=%d wei=%d bias_align=%d", $time, $signed(image_in[0][0][0]), $signed(patch_w[0][0][0][0]), 
                                         $signed((($signed(patch_b[0]) * $signed(128'(dyadic_params::patch_embed_bias_m))) >>> dyadic_params::patch_embed_bias_s)));
                            end
                            pixel_cnt <= pixel_cnt + 1;
                        end else begin 
                            for(int d=0; d<256; d++) x[patch_cnt][d] <= patch_acc[d]; 
                            if (patch_cnt == 0) begin
                                $display("[%0t]   [PATCH_EMBED Trace] Token 0 Output[0:3]: %d %d %d %d", $time, $signed(patch_acc[0]), $signed(patch_acc[1]), $signed(patch_acc[2]), $signed(patch_acc[3]));
                                $display("[%0t]   Expected:                                1616 1479 679 -28829", $time);
                            end
                            pixel_cnt <= 0;
                            patch_cnt <= patch_cnt + 1;
                        end
                    end else begin
                        state <= ST_POS_EMBED;
                        patch_cnt <= 0;
                    end
                end
                
                ST_POS_EMBED: begin
                    $display("[%0t] === POSITION EMBEDDING ===", $time);
                    
                    for(int i=0; i<64; i++) for(int j=0; j<256; j++) begin
                        // Scale patch projection output
                        automatic logic signed [127:0] term1 = ($signed(x[i][j]) * $signed(128'(dyadic_params::patch_proj_align_m))) >>> dyadic_params::patch_proj_align_s;
                        // Scale position embedding
                        automatic logic signed [127:0] term2 = ($signed(pos_embed[i][j]) * $signed(128'(dyadic_params::pos_embed_align_m))) >>> dyadic_params::pos_embed_align_s;
                        x[i][j] <= term1[63:0] + term2[63:0];
                    end
                    state <= ST_POS_EMBED_WAIT;
                end
                
                ST_POS_EMBED_WAIT: begin
                    $display("[%0t] === POSITION EMBEDDING COMPLETE ===", $time);
                    begin
                        automatic logic signed [63:0] x_mean = 0;
                        for (int i=0; i<256; i++) x_mean = x_mean + $signed(x[0][i]);
                        x_mean = x_mean / 256;
                        $display("[%0t] x[0][0:3] AFTER pos_embed: %d %d %d %d | Mean: %d", 
                                 $time, $signed(x[0][0]), $signed(x[0][1]), $signed(x[0][2]), $signed(x[0][3]), x_mean);
                        $display("[%0t] Expected:                  7371 -75635 39137 59478 | Mean: 39", $time);
                    end
                    b_curr <= 0; patch_cnt <= 0; state <= ST_BLOCK_START;
                end
                
                ST_BLOCK_START: begin
                    $display("[%0t] === EXECUTING BLOCK %0d START ===", $time, b_curr);
                    state <= ST_LN1;
                end
                
                ST_LN1: begin
                    if (patch_cnt < 64) begin
                        if (!ln_start && !ln_done) begin 
                            ln_start <= 1; 
                            ln_in <= x[patch_cnt]; 
                            ln_w <= b_norm1_w[b_curr]; 
                            
                            // CRITICAL FIX: Apply bias alignment correctly
                            for (int d=0; d<256; d++) begin
                                automatic logic signed [127:0] bias_scaled = ($signed(b_norm1_b[b_curr][d]) * $signed(128'(dyadic_params::n1_bias_align_m[b_curr]))) >>> dyadic_params::n1_bias_align_s[b_curr];
                                ln_b[d] <= bias_scaled[31:0];
                            end
                            
                            ln_m_fixed <= dyadic_params::n1_mult_m[b_curr]; 
                            ln_s_fixed <= dyadic_params::n1_mult_s[b_curr];
                        end else if (ln_done) begin
                            ln_start <= 0; 
                            
                            // CRITICAL FIX: Apply proper scaling to int8 with saturation
                            for(int d=0; d<256; d++) begin
                                automatic logic signed [127:0] scaled = ($signed(ln_out[d]) * $signed(128'(dyadic_params::xn1_to_int8_m[b_curr]))) >>> dyadic_params::xn1_to_int8_s[b_curr]; 
                                if ($signed(scaled) > $signed(128'sd127)) mmu1_a[patch_cnt][d] <= 8'sd127;
                                else if ($signed(scaled) < $signed(-128'sd128)) mmu1_a[patch_cnt][d] <= -8'sd128;
                                else mmu1_a[patch_cnt][d] <= scaled[7:0];
                            end
                            
                            if (patch_cnt == 0) begin
                                automatic logic signed [127:0] s0 = ($signed(ln_out[0]) * $signed(128'(dyadic_params::xn1_to_int8_m[b_curr]))) >>> dyadic_params::xn1_to_int8_s[b_curr];
                                automatic logic signed [127:0] s1 = ($signed(ln_out[1]) * $signed(128'(dyadic_params::xn1_to_int8_m[b_curr]))) >>> dyadic_params::xn1_to_int8_s[b_curr];
                                $display("[%0t] === BLOCK %0d LN1 OUTPUT ===", $time, b_curr);
                                $display("[%0t] ln_out[0][0:1]: %d %d", $time, $signed(ln_out[0]), $signed(ln_out[1]));
                                $display("[%0t] mmu1_a[0][0:1]: %d %d (Scheduled INT8)", $time, $signed(s0[7:0]), $signed(s1[7:0]));
                            end
                            
                            if (patch_cnt < 63) begin 
                                patch_cnt <= patch_cnt + 1; 
                            end else begin 
                                state <= ST_QKV; 
                                patch_cnt <= 0; 
                            end
                        end
                    end
                end
                
                ST_QKV: begin
                    // Tile over output columns: 768 cols = 12 tiles of 64.
                    // patch_cnt==99 is a sentinel meaning "buffer fully written, do reshape next cycle".
                    if (patch_cnt == 99) begin
                        // One cycle after all tiles written: QKV ping-pong read bank now holds complete data.
                        patch_cnt <= 0;
                        for (int p=0; p<64; p++) for (int qkv_idx=0; qkv_idx<3; qkv_idx++)
                            for (int h=0; h<8; h++) for (int d=0; d<32; d++) begin
                                automatic int flat_idx = qkv_idx * 256 + h * 32 + d;
                                automatic logic signed [127:0] bias_scaled = ($signed(b_qkv_b[b_curr][flat_idx]) * $signed(128'(dyadic_params::qkv_bias_m[b_curr]))) >>> dyadic_params::qkv_bias_s[b_curr];
                                // Keep ping-pong as staging only, but consume the legacy mmu1_buf path unchanged
                                // so Q/K/V ordering matches the original proven design.
                                automatic logic signed [63:0] val = mmu1_buf[p][flat_idx] + bias_scaled[63:0];

                                if ($signed(val) > $signed(64'sd2147483647)) begin
                                    if (qkv_idx == 0) q[h][p][d] <= 32'sd2147483647;
                                    else if (qkv_idx == 1) k[h][p][d] <= 32'sd2147483647;
                                    else v[h][p][d] <= 32'sd2147483647;
                                end else if ($signed(val) < $signed(-64'sd2147483648)) begin
                                    if (qkv_idx == 0) q[h][p][d] <= -32'sd2147483648;
                                    else if (qkv_idx == 1) k[h][p][d] <= -32'sd2147483648;
                                    else v[h][p][d] <= -32'sd2147483648;
                                end else begin
                                    if (qkv_idx == 0) q[h][p][d] <= val[31:0];
                                    else if (qkv_idx == 1) k[h][p][d] <= val[31:0];
                                    else v[h][p][d] <= val[31:0];
                                end
                            end

                        $display("[%0t]   [B%0d QKV Trace] Token 0 Output[0:3]: q=%0d k=%0d v=%0d", $time, b_curr,
                                 $signed(q[0][0][0]), $signed(k[0][0][0]), $signed(v[0][0][0]));
                        $display("[%0t]   Expected:                                q=18186 k=777 v=-3036", $time);

                        state <= ST_SOFT_SCORES;
                        head_cnt <= 0; patch_cnt_j <= 0;

                    end else if (!mmu_start && !mmu_done) begin
                        if (patch_cnt < 2) begin
                            patch_cnt <= patch_cnt + 1;
                        end else begin
                            mmu_start <= 1;
                            tile_dim <= 16'd256; 
                            patch_cnt <= 0;
                            if (col_tile == 0)
                                $display("[%0t] Starting QKV MMU tile %0d/12. mmu1_a[0][0:3] = %d %d %d %d", $time,
                                         col_tile, $signed(mmu1_a[0][0]), $signed(mmu1_a[0][1]),
                                         $signed(mmu1_a[0][2]), $signed(mmu1_a[0][3]));
                        end
                    end else if (mmu_done) begin
                        mmu_start <= 0;

                        // Write this 64-col tile into the generic ping-pong buffer.
                        // Keep the legacy mirror write too for debug compatibility.
                        qkv_buf_wr_en <= 1;
                        for (int i = 0; i < 64; i++)
                            for (int j = 0; j < 64; j++)
                                mmu1_buf[i][col_tile * 64 + j] <= mmu_out[i][j];

                        if (col_tile < 11) begin
                            col_tile <= col_tile + 1;
                        end else begin
                            // Final tile for this block: publish freshly written bank for readout next cycle.
                            qkv_buf_swap <= 1;
                            col_tile <= 0;
                            patch_cnt <= 99;  // sentinel: trigger reshape next cycle
                        end
                    end
                end
                
                ST_SOFT_SCORES: begin
                    // Compute attention scores for one head at a time
                    if (head_cnt < 8) begin
                        if (patch_cnt < 64) begin
                            if (patch_cnt_j == 0 && !soft_start && !soft_done) begin
                                // Initialize row_scores for this query token
                                for(int j=0; j<64; j++) begin
                                    automatic logic signed [63:0] acc = 0;
                                    for(int d=0; d<32; d++) begin
                                        acc = acc + (signed'(128'(q[head_cnt][patch_cnt][d])) * signed'(128'(k[head_cnt][j][d])));
                                    end
                                    row_scores[j] <= acc;
                                    if (head_cnt == 0 && patch_cnt == 0 && j < 4) begin
                                        $display("[%0t]   [B%0d ATTN Trace] H0 Q0 K%0d: q[0]=%d k[0]=%d acc=%d", $time, b_curr, j, 
                                                 $signed(q[0][0][0]), $signed(k[0][j][0]), acc);
                                    end
                                end
                                patch_cnt_j <= 64; // Mark as computed
                            end else if (patch_cnt_j == 64 && !soft_start && !soft_done) begin
                                // Launch softmax for this row
                                soft_in <= row_scores;
                                soft_m_idx <= dyadic_params::soft_idx_m[b_curr];
                                soft_s_idx <= dyadic_params::soft_idx_s[b_curr];
                                
                                $display("[%0t] Block %0d Softmax start. m_idx=%0d, s_idx=%0d", $time, b_curr, dyadic_params::soft_idx_m[b_curr], dyadic_params::soft_idx_s[b_curr]);
                                
                                soft_start <= 1;
                                state <= ST_SOFTMAX;
                            end
                        end else begin
                            // All patches done for this head
                            head_cnt <= head_cnt + 1;
                            patch_cnt <= 0;
                            patch_cnt_j <= 0;
                            if (head_cnt >= 7) begin
                                $display("[%0t] Block %0d All Heads Done. V[0,0,0]=%d, V[7,63,31]=%d", $time, b_curr, $signed(v[0][0][0]), $signed(v[7][63][31]));
                                state <= ST_ATTN_V;
                            end
                        end
                    end
                end
                
                ST_SOFTMAX: begin
                    if (soft_done) begin
                        soft_start <= 0;
                        
                        for(int j=0; j<64; j++) attn_weights[head_cnt][patch_cnt][j] <= soft_out[j];
                        
                        begin
                            automatic logic signed [63:0] cur_max = -64'h7FFFFFFFFFFFFFFF;
                            for(int k=0; k<64; k++) if(soft_in[k] > cur_max) cur_max = soft_in[k];
                            $display("[%0t] Block %0d Softmax[%0d][%0d] Trace: Score0=%d, Max=%d", 
                                     $time, b_curr, head_cnt, patch_cnt, $signed(soft_in[0]), cur_max);
                        end
                        $display("[%0t] Block %0d Softmax[%0d][%0d][0:3]: %d %d %d %d", $time, b_curr, head_cnt, patch_cnt, 
                                 $signed(8'(soft_out[0])), $signed(8'(soft_out[1])), $signed(8'(soft_out[2])), $signed(8'(soft_out[3])));
                        
                        if (head_cnt == 0 && patch_cnt == 0) $display("[%0t] Block %0d Weights0[0,0]=%d", $time, b_curr, $signed(8'(soft_out[0])));
                        
                        patch_cnt <= patch_cnt + 1;
                        patch_cnt_j <= 0;
                        state <= ST_SOFT_SCORES;
                    end
                end
                
                ST_ATTN_V: begin
                    // Compute attention @ V for all heads
                    // Output: [64, 8*32] = [64, 256]
                    for(int h=0; h<8; h++) for(int i=0; i<64; i++) for(int d=0; d<32; d++) begin
                        automatic logic signed [63:0] local_acc = 0;
                        for(int j=0; j<64; j++) begin
                            local_acc = local_acc + ($signed(64'(attn_weights[h][i][j])) * $signed(64'(v[h][j][d])));
                        end
                        
                        begin
                            automatic logic signed [127:0] scaled128 = ($signed(local_acc) * $signed(128'(dyadic_params::xa_to_int8_m[b_curr]))) >>> dyadic_params::xa_to_int8_s[b_curr];
                            if ($signed(scaled128) > $signed(128'sd127)) attn_out[i][h*32 + d] <= 8'sd127;
                            else if ($signed(scaled128) < $signed(-128'sd128)) attn_out[i][h*32 + d] <= -8'sd128;
                            else attn_out[i][h*32 + d] <= scaled128[7:0];
                        end
                    end
                    $display("[%0t] Block %0d ST_ATTN_V: W[4,19,0]=%d, V[4,0,22]=%d, Out[19,150]=%d", 
                             $time, b_curr, $signed(attn_weights[4][19][0]), $signed(v[4][0][22]), $signed(attn_out[19][150]));
                    state <= ST_PROJ;
                end
                
                ST_PROJ: begin
                    // Tile over 256 output columns = 4 tiles of 64.
                    // patch_cnt==99: buffer fully written, apply bias next cycle.
                    if (patch_cnt == 99) begin
                        patch_cnt <= 0;
                        col_tile <= 0;
                        for (int i=0; i<64; i++) for (int j=0; j<256; j++) begin
                            automatic logic signed [127:0] bias_scaled = ($signed(b_atn_p_b[b_curr][j]) * $signed(128'(dyadic_params::proj_bias_m[b_curr]))) >>> dyadic_params::proj_bias_s[b_curr];
                            mmu1_p_buf[i][j] <= mmu1_p_buf[i][j] + bias_scaled[63:0];
                        end
                        state <= ST_RES1;
                    end else if (!mmu_start && !mmu_done) begin
                        $display("[%0t] Block %0d ST_PROJ tile %0d: Weight0[0,0]=%d, A[0,0]=%d, M=%d, S=%d",
                                 $time, b_curr, col_tile, $signed(b_atn_p_w[b_curr][0][0]),
                                 $signed(attn_out[0][0]), dyadic_params::proj_bias_m[b_curr], dyadic_params::proj_bias_s[b_curr]);
                        if (patch_cnt < 2) patch_cnt <= patch_cnt + 1;
                        else begin
                            mmu_start <= 1;
                            tile_dim <= 16'd256;
                            patch_cnt <= 0;
                        end
                    end else if (mmu_done) begin
                        mmu_start <= 0;
                        // Store tile into wide buffer
                        for (int i = 0; i < 64; i++)
                            for (int j = 0; j < 64; j++)
                                mmu1_p_buf[i][col_tile * 64 + j] <= mmu_out[i][j];

                        if (col_tile < 3) begin
                            col_tile <= col_tile + 1;
                        end else begin
                            // All 4 tiles written — wait one cycle then bias-add
                            patch_cnt <= 99;
                        end
                    end
                end
                
                ST_RES1: begin
                    for(int i=0; i<64; i++) for(int j=0; j<256; j++) begin
                        automatic logic signed [127:0] scaled = ($signed(mmu1_p_buf[i][j]) * $signed(128'(dyadic_params::res1_proj_align_m[b_curr]))) >>> dyadic_params::res1_proj_align_s[b_curr];
                        x[i][j] <= x[i][j] + scaled[63:0];
                    end
                    state <= ST_LN2;
                    patch_cnt <= 0;
                end
                
                ST_LN2: begin
                    if (patch_cnt < 64) begin
                        if (!ln_start && !ln_done) begin 
                            ln_start <= 1; 
                            ln_in <= x[patch_cnt]; 
                            ln_w <= b_norm2_w[b_curr]; 
                            
                            for (int d=0; d<256; d++) begin
                                automatic logic signed [127:0] bias_scaled = ($signed(b_norm2_b[b_curr][d]) * $signed(128'(dyadic_params::n2_bias_align_m[b_curr]))) >>> dyadic_params::n2_bias_align_s[b_curr];
                                ln_b[d] <= bias_scaled[31:0];
                            end
                            
                            ln_m_fixed <= dyadic_params::n2_mult_m[b_curr]; 
                            ln_s_fixed <= dyadic_params::n2_mult_s[b_curr];
                        end else if (ln_done) begin
                            ln_start <= 0; 
                            
                            for(int d=0; d<256; d++) begin
                                automatic logic signed [127:0] scaled128 = ($signed(ln_out[d]) * $signed(128'(dyadic_params::xn2_to_int8_m[b_curr]))) >>> dyadic_params::xn2_to_int8_s[b_curr]; 
                                if ($signed(scaled128) > $signed(128'sd127)) mmu2_a[patch_cnt][d] <= 8'sd127;
                                else if ($signed(scaled128) < $signed(-128'sd128)) mmu2_a[patch_cnt][d] <= -8'sd128;
                                else mmu2_a[patch_cnt][d] <= scaled128[7:0];
                            end
                            
                            if (patch_cnt < 63) begin 
                                patch_cnt <= patch_cnt + 1; 
                            end else begin 
                                state <= ST_MLP1; 
                                patch_cnt <= 0; 
                            end
                        end
                    end
                end
                
                ST_MLP1: begin
                    // Tile over 768 output columns = 12 tiles of 64.
                    // patch_cnt==99: buffer fully written, apply bias+gelu-in next cycle.
                    if (patch_cnt == 99) begin
                        patch_cnt <= 0;
                        col_tile <= 0;
                        for (int i=0; i<64; i++) for (int d=0; d<768; d++) begin
                            automatic logic signed [127:0] bias_scaled = ($signed(b_mlp1_b[b_curr][d]) * $signed(128'(dyadic_params::fc1_bias_m[b_curr]))) >>> dyadic_params::fc1_bias_s[b_curr];
                            gelu_in[i][d] <= mmu2_buf[i][d] + bias_scaled[63:0];
                        end
                        state <= ST_GELU;
                    end else if (!mmu_start && !mmu_done) begin
                        if (patch_cnt < 2) patch_cnt <= patch_cnt + 1;
                        else begin
                            mmu_start <= 1;
                            tile_dim <= 16'd256;
                            patch_cnt <= 0;
                        end
                    end else if (mmu_done) begin
                        mmu_start <= 0;
                        // Mirror this 64-col tile into the MLP1 ping-pong buffer.
                        // Keep the legacy wide buffer write unchanged for the verified datapath.
                        mlp1_buf_wr_en <= 1;
                        // Store tile into wide buffer
                        for (int i = 0; i < 64; i++)
                            for (int j = 0; j < 64; j++)
                                mmu2_buf[i][col_tile * 64 + j] <= mmu_out[i][j];

                        if (col_tile < 11) begin
                            col_tile <= col_tile + 1;
                        end else begin
                            // Final tile mirrored; publish freshly written bank while legacy path remains unchanged.
                            mlp1_buf_swap <= 1;
                            // All 12 tiles written — wait one cycle then bias+gelu
                            patch_cnt <= 99;
                        end
                    end
                end
                
                ST_GELU: begin
                    if (!gelu_start && !gelu_done) begin
                        gelu_m_idx <= dyadic_params::gelu_idx_m[b_curr]; 
                        gelu_s_idx <= dyadic_params::gelu_idx_s[b_curr];
                        gelu_start <= 1;
                    end else if (gelu_done) begin
                        gelu_start <= 0; 
                        
                        for(int i=0; i<64; i++) for(int d=0; d<768; d++) begin
                            automatic logic signed [127:0] scaled128 = ($signed(gelu_out[i][d]) * $signed(128'(dyadic_params::mlp_act_to_int8_m[b_curr]))) >>> dyadic_params::mlp_act_to_int8_s[b_curr];
                            if ($signed(scaled128) > $signed(128'sd127)) mmu2_p_a[i][d] <= 8'sd127;
                            else if ($signed(scaled128) < $signed(-128'sd128)) mmu2_p_a[i][d] <= -8'sd128;
                            else mmu2_p_a[i][d] <= scaled128[7:0];
                        end
                        state <= ST_MLP2;
                    end
                end
                
                ST_MLP2: begin
                    // Tile over 256 output columns = 4 tiles of 64.
                    // patch_cnt==99: buffer fully written, apply bias next cycle.
                    if (patch_cnt == 99) begin
                        patch_cnt <= 0;
                        col_tile <= 0;
                        for (int i=0; i<64; i++) for (int j=0; j<256; j++) begin
                            automatic logic signed [127:0] bias_scaled = ($signed(b_mlp2_b[b_curr][j]) * $signed(128'(dyadic_params::fc2_bias_m[b_curr]))) >>> dyadic_params::fc2_bias_s[b_curr];
                            mmu2_p_buf[i][j] <= mmu2_p_buf[i][j] + bias_scaled[63:0];
                        end
                        state <= ST_RES2;
                    end else if (!mmu_start && !mmu_done) begin
                        if (patch_cnt < 2) patch_cnt <= patch_cnt + 1;
                        else begin
                            mmu_start <= 1;
                            tile_dim <= 16'd768;
                            patch_cnt <= 0;
                        end
                    end else if (mmu_done) begin
                        mmu_start <= 0;
                        // Mirror this 64-col tile into the MLP2 ping-pong buffer.
                        // Keep the legacy wide buffer write unchanged for the verified datapath.
                        mlp2_buf_wr_en <= 1;
                        // Store tile into wide buffer
                        for (int i = 0; i < 64; i++)
                            for (int j = 0; j < 64; j++)
                                mmu2_p_buf[i][col_tile * 64 + j] <= mmu_out[i][j];

                        if (col_tile < 3) begin
                            col_tile <= col_tile + 1;
                        end else begin
                            // Final tile mirrored; publish freshly written bank while legacy path remains unchanged.
                            mlp2_buf_swap <= 1;
                            // All 4 tiles written — wait one cycle then bias-add
                            patch_cnt <= 99;
                        end
                    end
                end
                
                ST_RES2: begin
                    for(int i=0; i<64; i++) for(int j=0; j<256; j++) begin
                        automatic logic signed [127:0] scaled = ($signed(mmu2_p_buf[i][j]) * $signed(128'(dyadic_params::res2_mlp_align_m[b_curr]))) >>> dyadic_params::res2_mlp_align_s[b_curr];
                        x[i][j] <= x[i][j] + scaled[63:0];
                    end
                    begin
                        automatic logic signed [63:0] b_mean = 0;
                        for (int i=0; i<256; i++) b_mean = b_mean + $signed(x[0][i]);
                        $display("[%0t]   [BLOCK_%0d] COMPLETED. Token 0 Mean: %d", $time, b_curr, b_mean/256);
                    end
                    state <= ST_BLOCK_NEXT;
                end
                
                ST_BLOCK_NEXT: begin
                    if (b_curr < 5) begin 
                        b_curr <= b_curr + 1; 
                        patch_cnt <= 0; // CRITICAL: Reset for next block
                        state <= ST_BLOCK_START; 
                    end else begin 
                        state <= ST_NORM_FINAL; 
                        patch_cnt <= 0; 
                    end
                end
                
                ST_NORM_FINAL: begin
                    if (patch_cnt < 64) begin
                        if (!ln_start && !ln_done) begin 
                            ln_start <= 1; 
                            ln_in <= x[patch_cnt]; 
                            ln_w <= norm_f_w; 
                            
                            for (int d=0; d<256; d++) begin
                                automatic logic signed [127:0] bias_scaled = ($signed(norm_f_b[d]) * $signed(128'(dyadic_params::final_norm_bias_align_m))) >>> dyadic_params::final_norm_bias_align_s;
                                ln_b[d] <= bias_scaled[31:0];
                            end
                            
                            ln_m_fixed <= dyadic_params::final_norm_mult_m; 
                            ln_s_fixed <= dyadic_params::final_norm_mult_s;
                        end else if (ln_done) begin 
                            ln_start <= 0; 
                            for(int d=0; d<256; d++) x[patch_cnt][d] <= $signed(64'(ln_out[d]));
                            patch_cnt <= patch_cnt + 1;
                        end
                    end else begin 
                        state <= ST_GAP;
                        patch_cnt <= 0;
                    end
                end
                
                ST_GAP: begin
                    // Global Average Pooling
                    for(int d=0; d<256; d++) begin
                        automatic logic signed [63:0] sum = 0;
                        for(int i=0; i<64; i++) sum = sum + x[i][d];
                        patch_acc[d] <= sum >>> 6; // Divide by 64
                    end
                    state <= ST_HEAD;
                end
                
                 ST_HEAD: begin
                    automatic logic signed [63:0] max_l = -64'h7BFFFFFFFFFFFFFF; 
                    automatic logic signed [63:0] second_l = -64'h7BFFFFFFFFFFFFFF;
                    automatic int m_i = 0;
                    automatic int sat_cnt = 0;
                    
                    $display("[%0t] === HEAD CALCULATION START ===", $time);
                    $display("[%0t]   GAP[0:3]: %d %d %d %d", $time, $signed(patch_acc[0]), $signed(patch_acc[1]), $signed(patch_acc[2]), $signed(patch_acc[3]));

                    // Classification head
                    for(int c=0; c<10; c++) begin
                        automatic logic signed [127:0] bias_scaled_128 = ($signed(head_b[c]) * $signed(128'(dyadic_params::head_bias_m))) >>> dyadic_params::head_bias_s;
                        automatic logic signed [63:0] acc = bias_scaled_128[63:0];
                        if (c < 4) $display("[%0t]   Head Bias[%0d] Aligned: %d", $time, c, acc);
                        
                        for(int d=0; d<256; d++) begin
                            automatic logic signed [127:0] gap_scaled_128 = ($signed(patch_acc[d]) * $signed(128'(dyadic_params::head_to_int8_m))) >>> dyadic_params::head_to_int8_s;
                            automatic logic signed [7:0] gap_int8;
                            
                            if ($signed(gap_scaled_128) > $signed(128'sd127)) begin gap_int8 = 8'sd127; if (c==0) sat_cnt++; end
                            else if ($signed(gap_scaled_128) < $signed(-128'sd128)) begin gap_int8 = -8'sd128; if (c==0) sat_cnt++; end
                            else gap_int8 = gap_scaled_128[7:0];
                            
                            acc = acc + ($signed(64'(gap_int8)) * $signed(64'(head_w[c][d])));
                        end
                        
                        if (acc > max_l) begin 
                            second_l = max_l;
                            max_l = acc; 
                            m_i = c; 
                        end else if (acc > second_l) begin
                            second_l = acc;
                        end
                        $display("[%0t] Logit[%0d] = %d", $time, c, acc);
                    end
                    
                    $display("[%0t] GAP Saturation Count: %0d/256", $time, sat_cnt);
                    $display("[%0t] PREDICTION: %0d | Confidence Margin: %d", $time, m_i, max_l - second_l);
                    prediction <= m_i; 
                    state <= ST_DONE;
                end
                
                ST_DONE: begin 
                    done <= 1; 
                    state <= ST_IDLE; 
                end
            endcase
        end
    end

endmodule