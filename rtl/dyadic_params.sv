// Auto-generated Dyadic Parameters for ViT
package dyadic_params;

    localparam logic signed [31:0] pos_embed_align_m = 32'd543272168;
    localparam logic [31:0]        pos_embed_align_s = 32'd20;
    localparam logic signed [31:0] patch_proj_align_m = 32'd794251129;
    localparam logic [31:0]        patch_proj_align_s = 32'd28;
    localparam logic signed [31:0] patch_embed_bias_m = 32'd956615878;
    localparam logic [31:0]        patch_embed_bias_s = 32'd47;

    localparam logic signed [31:0] n1_bias_align_m [0:5] = '{
        32'd1050188911,
        32'd664278338,
        32'd537082841,
        32'd903393923,
        32'd640839469,
        32'd663169506
    };
    localparam logic [31:0] n1_bias_align_s [0:5] = '{
        32'd57,
        32'd57,
        32'd57,
        32'd57,
        32'd57,
        32'd58
    };

    localparam logic signed [31:0] n1_mult_m [0:5] = '{
        32'd802931050,
        32'd541285962,
        32'd952748621,
        32'd938712520,
        32'd611385413,
        32'd572483577
    };
    localparam logic [31:0] n1_mult_s [0:5] = '{
        32'd31,
        32'd31,
        32'd32,
        32'd32,
        32'd31,
        32'd31
    };

    localparam logic signed [31:0] xn1_to_int8_m [0:5] = '{
        32'd536870912,
        32'd536870912,
        32'd536870912,
        32'd536870912,
        32'd536870912,
        32'd536870912
    };
    localparam logic [31:0] xn1_to_int8_s [0:5] = '{
        32'd29,
        32'd29,
        32'd29,
        32'd29,
        32'd29,
        32'd29
    };

    localparam logic signed [31:0] qkv_bias_m [0:5] = '{
        32'd587888925,
        32'd996486561,
        32'd775374405,
        32'd1055947626,
        32'd1065539177,
        32'd930164935
    };
    localparam logic [31:0] qkv_bias_s [0:5] = '{
        32'd45,
        32'd47,
        32'd47,
        32'd48,
        32'd48,
        32'd48
    };

    localparam logic signed [31:0] xa_to_int8_m [0:5] = '{
        32'd673497544,
        32'd700362518,
        32'd860498581,
        32'd768747363,
        32'd975225553,
        32'd991712998
    };
    localparam logic [31:0] xa_to_int8_s [0:5] = '{
        32'd45,
        32'd45,
        32'd45,
        32'd45,
        32'd45,
        32'd45
    };

    localparam logic signed [31:0] proj_bias_m [0:5] = '{
        32'd893190178,
        32'd702546275,
        32'd637805821,
        32'd797932392,
        32'd627532409,
        32'd1017162510
    };
    localparam logic [31:0] proj_bias_s [0:5] = '{
        32'd47,
        32'd48,
        32'd48,
        32'd49,
        32'd48,
        32'd49
    };

    localparam logic signed [31:0] res1_proj_align_m [0:5] = '{
        32'd576634742,
        32'd564230088,
        32'd1056002079,
        32'd661121553,
        32'd974273633,
        32'd897971340
    };
    localparam logic [31:0] res1_proj_align_s [0:5] = '{
        32'd29,
        32'd28,
        32'd29,
        32'd28,
        32'd29,
        32'd28
    };

    localparam logic signed [31:0] n2_bias_align_m [0:5] = '{
        32'd804541271,
        32'd670542330,
        32'd1044748771,
        32'd851843784,
        32'd587382701,
        32'd1020156329
    };
    localparam logic [31:0] n2_bias_align_s [0:5] = '{
        32'd57,
        32'd57,
        32'd57,
        32'd57,
        32'd56,
        32'd57
    };

    localparam logic signed [31:0] n2_mult_m [0:5] = '{
        32'd761970312,
        32'd820779257,
        32'd977312335,
        32'd1040821357,
        32'd554797579,
        32'd556781731
    };
    localparam logic [31:0] n2_mult_s [0:5] = '{
        32'd31,
        32'd32,
        32'd32,
        32'd32,
        32'd31,
        32'd31
    };

    localparam logic signed [31:0] xn2_to_int8_m [0:5] = '{
        32'd536870912,
        32'd536870912,
        32'd536870912,
        32'd536870912,
        32'd536870912,
        32'd536870912
    };
    localparam logic [31:0] xn2_to_int8_s [0:5] = '{
        32'd29,
        32'd29,
        32'd29,
        32'd29,
        32'd29,
        32'd29
    };

    localparam logic signed [31:0] fc1_bias_m [0:5] = '{
        32'd643958453,
        32'd1037507097,
        32'd552376674,
        32'd883881260,
        32'd1053074292,
        32'd592808074
    };
    localparam logic [31:0] fc1_bias_s [0:5] = '{
        32'd48,
        32'd50,
        32'd49,
        32'd50,
        32'd50,
        32'd49
    };

    localparam logic signed [31:0] mlp_act_to_int8_m [0:5] = '{
        32'd702215219,
        32'd956575905,
        32'd800627093,
        32'd840016618,
        32'd793681646,
        32'd1070740123
    };
    localparam logic [31:0] mlp_act_to_int8_s [0:5] = '{
        32'd48,
        32'd48,
        32'd48,
        32'd48,
        32'd48,
        32'd50
    };

    localparam logic signed [31:0] fc2_bias_m [0:5] = '{
        32'd740156261,
        32'd999731187,
        32'd703817044,
        32'd793598790,
        32'd875450360,
        32'd756892020
    };
    localparam logic [31:0] fc2_bias_s [0:5] = '{
        32'd50,
        32'd50,
        32'd50,
        32'd50,
        32'd51,
        32'd52
    };

    localparam logic signed [31:0] res2_mlp_align_m [0:5] = '{
        32'd615626279,
        32'd593934063,
        32'd600998055,
        32'd623075750,
        32'd686430541,
        32'd622523304
    };
    localparam logic [31:0] res2_mlp_align_s [0:5] = '{
        32'd26,
        32'd27,
        32'd27,
        32'd27,
        32'd27,
        32'd23
    };

    localparam logic signed [31:0] soft_idx_m [0:5] = '{
        32'd564999799,
        32'd1010394708,
        32'd577344863,
        32'd1023317548,
        32'd572864141,
        32'd849149154
    };
    localparam logic [31:0] soft_idx_s [0:5] = '{
        32'd57,
        32'd56,
        32'd55,
        32'd55,
        32'd54,
        32'd54
    };

    localparam logic signed [31:0] gelu_idx_m [0:5] = '{
        32'd870842798,
        32'd560712487,
        32'd640267511,
        32'd1069393339,
        32'd892994708,
        32'd733343331
    };
    localparam logic [31:0] gelu_idx_s [0:5] = '{
        32'd38,
        32'd36,
        32'd37,
        32'd38,
        32'd38,
        32'd37
    };

    localparam logic signed [31:0] head_to_int8_m = 32'd536870912;
    localparam logic [31:0]        head_to_int8_s = 32'd29;
    localparam logic signed [31:0] head_bias_m = 32'd648279411;
    localparam logic [31:0]        head_bias_s = 32'd50;

    localparam logic signed [31:0] final_norm_bias_align_m = 32'd670861798;
    localparam logic [31:0]        final_norm_bias_align_s = 32'd58;
    localparam logic signed [31:0] final_norm_mult_m = 32'd805876567;
    localparam logic [31:0]        final_norm_mult_s = 32'd30;

endpackage
