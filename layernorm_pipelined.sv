// High-Performance Pipelined LayerNorm (layernorm_pipelined.sv)
`timescale 1ns / 1ps

module layernorm_pipelined #(
    parameter int D = 256
    //parameter string INV_SQRT_LUT = "C:/Users/DELTA2/Downloads/vit_final/vit_new/export_quantized_new/inv_sqrt_lut.mem"
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [63:0] x_in [D],
    input  logic signed [7:0]  weight [D],
    input  logic signed [31:0] bias [D],
    input  logic signed [31:0] m_fixed,
    input  logic signed [31:0] s_fixed,
    output logic signed [31:0] x_out [D],
    output logic done
);

    import softmax_lut_pkg::*;
    
    // Internal States
    typedef enum logic [2:0] {IDLE, MEAN, VAR, NORM, DONE_STATE} state_t;
    state_t state;
    logic [15:0] cnt;
    
    // Calculation Registers
    logic signed [63:0] sum;
    logic signed [63:0] mean;
    logic signed [127:0] var_sum;
    logic signed [63:0] variance;
    logic signed [63:0] centered [D];

    // Per-patch normalization constants
    logic [6:0] blen;
    logic signed [6:0] shift_val;
    logic [63:0] norm_v;
    int lut_idx;
    logic signed [63:0] m_lut;
    logic signed [6:0] s_lut;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            cnt <= 0;
            sum <= 0;
            var_sum <= 0;
            m_lut <= 0;
            s_lut <= 0;
            for (int i=0; i<D; i++) x_out[i] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= MEAN;
                        cnt <= 0;
                        sum <= 0;
                        $display("[LN START] Weight[0]=%d, Bias[0]=%d, x_in[0]=%d", $signed(weight[0]), $signed(bias[0]), $signed(x_in[0]));
                    end
                end
                
                MEAN: begin
                    if (cnt < D) begin
                        sum <= sum + x_in[cnt];
                        cnt <= cnt + 1;
                    end else begin
                        mean <= sum >>> 8;
                        state <= VAR;
                        cnt <= 0;
                        var_sum <= 0;
                    end
                end
                
                VAR: begin
                    if (cnt < D) begin
                        automatic logic signed [63:0] c = x_in[cnt] - mean;
                        centered[cnt] <= c;
                        // Use full precision (k=0) for variance
                        var_sum <= var_sum + $signed(c * c);
                        cnt <= cnt + 1;
                    end else begin
                        variance <= var_sum >>> 8;
                        state <= NORM;
                        cnt <= 0;
                    end
                end
                
                NORM: begin
                    if (cnt < D) begin
                        if (cnt == 0) begin
                            // Python-equivalent hw_inv_sqrt
                            automatic logic [63:0] safe_var = (variance <= 0) ? 64'd1 : variance;
                            blen = 0;
                            for (int b=63; b>=0; b--) if (safe_var[b]) begin blen = b + 1; break; end
                            
                            if (blen % 2 == 0) shift_val = blen - 30;
                            else               shift_val = blen - 31;
                            
                            if (shift_val > 0) norm_v = safe_var >> shift_val;
                            else               norm_v = safe_var << (-shift_val);
                            
                            // v_min = 536870912 (0.5 * 2^30)
                            lut_idx = ($signed(128'(norm_v)) - $signed(128'd536870912)) * $signed(128'd1023) / $signed(128'd1610612736);
                            if (lut_idx < 0) lut_idx = 0; if (lut_idx > 1023) lut_idx = 1023;
                            
                            m_lut = $signed(64'(INV_SQRT_LUT[lut_idx]));
                            s_lut = 45 + (shift_val >>> 1);
                            
                            if (variance <= 0) begin m_lut = 64'd1073741824; s_lut = 0; end
                            
                        end
                        
                        // prod = centered * weight
                        begin
                            // CRITICAL FIX: Explicitly sign-extend to 128 bits before multiplication
                            automatic logic signed [127:0] s_centered = $signed(centered[cnt]);
                            automatic logic signed [127:0] s_weight = $signed(weight[cnt]);
                            automatic logic signed [127:0] prod128 = s_centered * s_weight;
                            
                            // term = (prod * m_lut) >> s_lut
                            automatic logic signed [127:0] s_m_lut = $signed(m_lut);
                            automatic logic signed [127:0] term128 = (prod128 * s_m_lut) >>> s_lut;
                            
                            // final_term = (term * m_fixed) >> s_fixed
                            automatic logic signed [127:0] s_m_fixed = $signed(m_fixed); // m_fixed is port input
                            automatic logic signed [127:0] final_term128 = (term128 * s_m_fixed) >>> s_fixed;
                            
                            // out = final_term + bias
                            automatic logic signed [127:0] s_bias = $signed(bias[cnt]);
                            automatic logic signed [127:0] out128 = final_term128 + s_bias;
                            
                            if ($signed(out128) > $signed(128'sd2147483647))      x_out[cnt] <= 32'sd2147483647;
                            else if ($signed(out128) < $signed(-128'sd2147483648)) x_out[cnt] <= -32'sd2147483648;
                            else                              x_out[cnt] <= out128[31:0];
                            
                        end
                        
                        cnt <= cnt + 1;
                    end else begin
                        state <= DONE_STATE;
                    end
                end
                
                DONE_STATE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
