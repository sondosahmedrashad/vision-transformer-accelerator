// High-Performance Pipelined GELU (gelu_pipelined.sv)
`timescale 1ns / 1ps

module gelu_pipelined #(
    parameter int N = 64,
    parameter int D = 768 // MLP dimension
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [63:0] x_in [N][D],
    input  logic signed [31:0] m_idx,
    input  logic signed [31:0] s_idx,
    output logic signed [31:0] x_out [N][D],
    output logic done
);

    typedef enum logic [1:0] {IDLE, COMPUTE, DONE_STATE} state_t;
    state_t state;
    logic [15:0] i_cnt, d_cnt;

    logic [8:0] lut_addr;
    logic signed [31:0] lut_data;

    // Instantiate LUT ROM
    gelu_lut lut_inst (
        .addr(lut_addr),
        .data(lut_data)
    );
    
    int idx;
    logic [8:0] lut_addr_r;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done  <= 0;
            i_cnt <= 0;
            d_cnt <= 0;
            lut_addr_r <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= COMPUTE;
                        i_cnt <= 0;
                        d_cnt <= 0;
                    end
                end

                COMPUTE: begin
                    if (i_cnt < N) begin
                        if (d_cnt < D) begin
                            //val128     = $signed(x_in[i_cnt][d_cnt]);
                            //idx_offset = int'((val128 * $signed(m_idx)) >>> s_idx);
                            

                            // write previous cycle LUT output
                            x_out[i_cnt][d_cnt] <= lut_data;

                            d_cnt <= d_cnt + 1;
                        end else begin
                            d_cnt <= 0;
                            i_cnt <= i_cnt + 1;
                        end
                    end else begin
                        state <= DONE_STATE;
                    end
                end

                DONE_STATE: begin
                    done  <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    always @(*) begin
        idx        = 256 + int'(($signed(x_in[i_cnt][d_cnt]) * $signed(m_idx)) >>> s_idx);
        if (idx < 0)   idx = 0;
        else if (idx > 511) idx = 511;

        lut_addr = idx[8:0];
    end

    //assign lut_addr = lut_addr_r;

endmodule
