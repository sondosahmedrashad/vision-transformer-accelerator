`timescale 1ns / 1ps

module matrix_ping_pong_buffer #(
    parameter int ROWS = 64,
    parameter int COLS = 768,
    parameter int TILE_COLS = 64,
    parameter int DW = 64,
    parameter int NUM_BLOCKS = 6,
    parameter int NUM_TILES = (COLS / TILE_COLS)
)(
    input  logic clk,
    input  logic rst,

    input  logic wr_en,
    input  logic [$clog2(NUM_BLOCKS)-1:0] wr_block_idx,
    input  logic [$clog2(NUM_TILES)-1:0]  wr_tile_idx,
    input  logic signed [DW-1:0] wr_tile [0:ROWS-1][0:TILE_COLS-1],

    input  logic swap_banks,

    input  logic [$clog2(NUM_BLOCKS)-1:0] rd_block_idx,
    output logic signed [DW-1:0] rd_matrix [0:ROWS-1][0:COLS-1],
    output logic rd_valid,
    output logic dbg_read_sel,
    output logic dbg_write_sel
);

    logic read_sel, write_sel;
    logic capture_pending;
    logic capture_sel;
    logic [$clog2(NUM_BLOCKS)-1:0] captured_block_idx;

    logic signed [DW-1:0] bank0 [0:NUM_BLOCKS-1][0:ROWS-1][0:COLS-1];
    logic signed [DW-1:0] bank1 [0:NUM_BLOCKS-1][0:ROWS-1][0:COLS-1];
    logic signed [DW-1:0] rd_matrix_reg [0:ROWS-1][0:COLS-1];

    integer b, i, j;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            read_sel          <= 1'b0;
            write_sel         <= 1'b0;   // start writing into bank0
            capture_pending   <= 1'b0;
            capture_sel       <= 1'b0;
            captured_block_idx<= '0;
            rd_valid          <= 1'b0;
            dbg_read_sel      <= 1'b0;
            dbg_write_sel     <= 1'b0;

            for (b = 0; b < NUM_BLOCKS; b = b + 1) begin
                for (i = 0; i < ROWS; i = i + 1) begin
                    for (j = 0; j < COLS; j = j + 1) begin
                        bank0[b][i][j] <= '0;
                        bank1[b][i][j] <= '0;
                    end
                end
            end

            for (i = 0; i < ROWS; i = i + 1) begin
                for (j = 0; j < COLS; j = j + 1) begin
                    rd_matrix_reg[i][j] <= '0;
                end
            end
        end else begin
            rd_valid <= 1'b0;
            dbg_read_sel  <= read_sel;
            dbg_write_sel <= write_sel;

            if (wr_en) begin
                for (i = 0; i < ROWS; i = i + 1) begin
                    for (j = 0; j < TILE_COLS; j = j + 1) begin
                        if (write_sel == 1'b0)
                            bank0[wr_block_idx][i][wr_tile_idx*TILE_COLS + j] <= wr_tile[i][j];
                        else
                            bank1[wr_block_idx][i][wr_tile_idx*TILE_COLS + j] <= wr_tile[i][j];
                    end
                end
            end

            // On swap: publish the just-completed write bank as the next read bank.
            // Do NOT capture rd_matrix_reg in the same cycle, because the final tile write
            // also lands in this edge. Instead, schedule a capture for the next cycle.
            if (swap_banks) begin
                read_sel           <= write_sel;
                write_sel          <= ~write_sel;
                capture_pending    <= 1'b1;
                capture_sel        <= write_sel;
                captured_block_idx <= rd_block_idx;
            end else if (capture_pending) begin
                for (i = 0; i < ROWS; i = i + 1) begin
                    for (j = 0; j < COLS; j = j + 1) begin
                        if (capture_sel == 1'b0)
                            rd_matrix_reg[i][j] <= bank0[captured_block_idx][i][j];
                        else
                            rd_matrix_reg[i][j] <= bank1[captured_block_idx][i][j];
                    end
                end
                rd_valid        <= 1'b1;   // one-cycle pulse after registered capture
                capture_pending <= 1'b0;
            end
        end
    end

    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                rd_matrix[r][c] = rd_matrix_reg[r][c];
            end
        end
    end

endmodule
