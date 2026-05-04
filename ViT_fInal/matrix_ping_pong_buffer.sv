`timescale 1ns / 1ps

module matrix_ping_pong_buffer #(
    parameter int ROWS = 64,
    parameter int COLS = 768,
    parameter int TILE_COLS = 64,
    parameter int DW = 64,
    parameter int NUM_TILES = (COLS / TILE_COLS),
    parameter int TILE_IDX_W = (NUM_TILES <= 1) ? 1 : $clog2(NUM_TILES)
)(
    input  logic clk,
    input  logic rst,
    input  logic wr_en,
    input  logic [TILE_IDX_W-1:0] wr_tile_idx,
    input  logic signed [DW-1:0] wr_tile [0:ROWS-1][0:TILE_COLS-1],
    input  logic swap_banks,
    output logic signed [DW-1:0] rd_matrix [0:ROWS-1][0:COLS-1]
);

    logic signed [DW-1:0] bank0 [0:ROWS-1][0:COLS-1];
    logic signed [DW-1:0] bank1 [0:ROWS-1][0:COLS-1];
    logic read_sel;
    logic write_sel;

    integer i, j;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            read_sel  <= 1'b0;
            write_sel <= 1'b1;
            for (i = 0; i < ROWS; i = i + 1)
                for (j = 0; j < COLS; j = j + 1) begin
                    bank0[i][j] <= '0;
                    bank1[i][j] <= '0;
                end
        end else begin
            if (wr_en) begin
                for (i = 0; i < ROWS; i = i + 1)
                    for (j = 0; j < TILE_COLS; j = j + 1) begin
                        if (write_sel == 1'b0)
                            bank0[i][wr_tile_idx * TILE_COLS + j] <= wr_tile[i][j];
                        else
                            bank1[i][wr_tile_idx * TILE_COLS + j] <= wr_tile[i][j];
                    end
            end

            if (swap_banks) begin
                read_sel  <= write_sel;
                write_sel <= read_sel;
            end
        end
    end

    always_comb begin
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                rd_matrix[r][c] = (read_sel == 1'b0) ? bank0[r][c] : bank1[r][c];
    end

endmodule
