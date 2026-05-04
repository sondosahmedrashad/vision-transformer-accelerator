`timescale 1ns/1ps

module softmax_pipelined #(
    parameter int N = 64
)(
    input  logic clk,
    input  logic rst,
    input  logic start,

    input  logic signed [63:0] x_in [N],
    input  logic signed [31:0] m_idx,
    input  logic signed [31:0] s_idx,

    output logic signed [7:0] x_out [N],
    output logic done
);

    import softmax_lut_pkg::*;

    typedef enum logic [2:0] {
        IDLE,
        MAX,
        EXP_SUM,
        RECIP,
        NORM,
        DONE_S
    } state_t;

    state_t state;

    logic [15:0] cnt;

    logic signed [63:0] max_val;
    logic signed [63:0] exp_val [N];
    logic signed [63:0] sum_exp;

    logic signed [63:0] recip_m;
    logic signed [31:0] recip_s;

    logic [6:0] blen;
    logic signed [6:0] shift_val;
    logic [63:0] norm_v;
    logic [10:0] lut_idx;

    logic signed [63:0] is_m;
    logic [6:0] is_s;

    logic signed [127:0] mult_tmp;
    logic signed [127:0] final_tmp;
    logic signed [127:0] result_tmp;

    //------------------------------------------
    // Bit-length (priority encoder)
    //------------------------------------------
    function automatic [6:0] bit_length(input logic [63:0] v);
        integer i;
        begin
            bit_length = 0;
            for (i = 63; i >= 0; i = i - 1)
                if (v[i] && bit_length == 0)
                    bit_length = i + 1;
        end
    endfunction

    //------------------------------------------
    // FSM
    //------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state   <= IDLE;
            done    <= 0;
            cnt     <= 0;
            sum_exp <= 0;
        end
        else begin
            case (state)

            //----------------------------------
            IDLE:
            begin
                done <= 0;
                if (start) begin
                    cnt     <= 0;
                    max_val <= -64'sh7FFFFFFFFFFFFFFF;
                    state   <= MAX;
                end
            end

            //----------------------------------
            MAX:
            begin
                if (cnt < N) begin
                    if (x_in[cnt] > max_val)
                        max_val <= x_in[cnt];
                    cnt <= cnt + 1;
                end
                else begin
                    cnt     <= 0;
                    sum_exp <= 0;
                    state   <= EXP_SUM;
                end
            end

            //----------------------------------
            EXP_SUM:
            begin
                if (cnt < N) begin

                    mult_tmp = (x_in[cnt] - max_val) * m_idx;
                    mult_tmp = mult_tmp >>> s_idx;

                    lut_idx = 1023 + mult_tmp[31:0];
                    if (lut_idx > 1023) lut_idx = 1023;

                    exp_val[cnt] <= EXP_LUT[lut_idx];
                    sum_exp      <= sum_exp + EXP_LUT[lut_idx];

                    cnt <= cnt + 1;
                end
                else begin
                    cnt   <= 0;
                    state <= RECIP;
                end
            end

            //----------------------------------
            RECIP:
            begin
                logic [63:0] safe_sum;

                safe_sum = (sum_exp == 0) ? 64'd1 : sum_exp;

                blen = bit_length(safe_sum);

                if (blen[0] == 0)
                    shift_val = blen - 30;
                else
                    shift_val = blen - 31;

                if (shift_val > 0)
                    norm_v = safe_sum >> shift_val;
                else
                    norm_v = safe_sum << (-shift_val);

                //----------------------------------
                // DIVISION REMOVED HERE
                //----------------------------------
                mult_tmp = (norm_v - 32'd536870912) * 1363;
                lut_idx  = mult_tmp >>> 31;

                if (lut_idx > 1023) lut_idx = 1023;

                is_m = INV_SQRT_LUT[lut_idx];
                is_s = 45 + (shift_val >>> 1);

                recip_m <= (is_m * is_m) >>> 30;
                recip_s <= (2 * is_s) - 30;

                cnt   <= 0;
                state <= NORM;
            end

            //----------------------------------
            NORM:
            begin
                if (cnt < N) begin

                    mult_tmp  = exp_val[cnt] * 127;
                    final_tmp = mult_tmp * recip_m;

                    if (recip_s >= 0)
                        result_tmp = final_tmp >>> recip_s;
                    else
                        result_tmp = final_tmp <<< (-recip_s);

                    if (result_tmp > 127)
                        x_out[cnt] <= 8'sd127;
                    else if (result_tmp < -128)
                        x_out[cnt] <= -8'sd128;
                    else
                        x_out[cnt] <= result_tmp[7:0];

                    cnt <= cnt + 1;
                end
                else begin
                    state <= DONE_S;
                end
            end

            //----------------------------------
            DONE_S:
            begin
                done  <= 1;
                state <= IDLE;
            end

            endcase
        end
    end

endmodule