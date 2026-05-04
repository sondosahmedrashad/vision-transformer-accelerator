vlib work
vlog vit_top_integrated.sv dyadic_params.sv softmax_lut_pkg.sv softmax_pipelined.sv gelu.sv gelu_pipelined.sv layernorm_pipelined.sv vit_tb.sv mmu_modular_complete.sv matrix_ping_pong_buffer.sv weight_rom.sv
vsim -voptargs=+acc work.vit_tb
add wave *
run -all
#quit -sim