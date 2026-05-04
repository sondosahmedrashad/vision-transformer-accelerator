// Comprehensive Testbench for Integrated High-Performance ViT (vit_tb.sv)
`timescale 1ns / 1ps

module vit_tb;

    // Clock and reset
    logic clk;
    logic rst;
    logic start;
    
    // Test inputs and outputs
    logic signed [7:0] image_in [0:2][0:63][0:63];
    logic done;
    logic [31:0] prediction;
    
    // CIFAR-10 class names
    string CIFAR10_CLASSES[0:9] = '{
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    
    // State names to match vit_top_integrated exactly (22 states)
    string state_names[0:21] = '{
        "IDLE", "PATCH_EMBED", "POS_EMBED", "POS_EMBED_WAIT",
        "BLOCK_START", "LN1", "QKV", "SOFT_SCORES", "SOFTMAX", "ATTN_V", "PROJ", "RES1",
        "LN2", "MLP1", "GELU", "MLP2", "RES2", "BLOCK_NEXT",
        "NORM_FINAL", "GAP", "HEAD", "DONE"
    };
    
    // Instantiate Integrated DUT
    vit_top_integrated dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .image_in(image_in),
        .done(done),
        .prediction(prediction)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Monitoring
    int cycle_count;
    always @(posedge clk or posedge rst) begin
        if (rst) cycle_count <= 0;
        else cycle_count <= cycle_count + 1;
    end
    
    logic [4:0] current_state;
    logic [4:0] prev_state;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= 0;
            prev_state <= 0;
        end else begin
            current_state <= dut.state;
            if (current_state != prev_state) begin
                $display("[%0t] State Transition: %s -> %s", $time, state_names[prev_state], state_names[current_state]);
                prev_state <= current_state;
            end

            // Progress Reporting
            if (current_state == 1) begin // PATCH_EMBED
                if (dut.pixel_cnt == 0 && dut.patch_cnt % 8 == 0 && dut.patch_cnt > 0)
                    $display("[%0t]   [PATCH_EMBED] Processed %0d/64 patches", $time, dut.patch_cnt);
            end
            
            if (current_state == 3) begin // BLOCK_START
                 $display("[%0t]   [BLOCK_%0d] Starting execution...", $time, dut.b_curr);
            end
            
            if (current_state == 15) begin // RES2
                 $display("[%0t]   [BLOCK_%0d] Completed.", $time, dut.b_curr);
            end
        end
    end

    // Test stimulus
    initial begin
        int image_file;
        logic signed [7:0] image_data;
        int status;
        string IMAGE_FILE = "C:/Users/user/Downloads/vit_new/real_images_new/test_17.mem";
        
        rst = 1; start = 0; #100;
        rst = 0; #100;
        
        $display("[%0t] Loading test image from: %s", $time, IMAGE_FILE);
        image_file = $fopen(IMAGE_FILE, "r");
        if (image_file) begin
            for (int c=0; c<3; c++) for (int i=0; i<64; i++) for (int j=0; j<64; j++) begin
                if (!$feof(image_file)) begin
                    status = $fscanf(image_file, "%h", image_data);
                    image_in[c][i][j] = image_data;
                end else begin
                    image_in[c][i][j] = 0;
                end
            end
            $fclose(image_file);
            $display("[%0t] Test image loaded successfully. image_in[0][0][0] = %d", $time, $signed(image_in[0][0][0]));
        end else begin
            $display("[%0t] ERROR: Could not open test image file: %s", $time, IMAGE_FILE);
            $finish;
        end
        
        #100;
        $display("[%0t] Starting Vision Transformer inference...", $time);
        start = 1; #10; start = 0;
        
        wait(done);
        
        $display("");
        $display("==========================================================");
        $display("INFERENCE COMPLETE");
        $display("==========================================================");
        $display("Final Results:");
        $display("  - Predicted Class: %0d (%s)", prediction, CIFAR10_CLASSES[prediction]);
        $display("  - Total Cycles: %0d", cycle_count);
        $display("==========================================================");
        
        #1000;
        $finish;
    end

endmodule