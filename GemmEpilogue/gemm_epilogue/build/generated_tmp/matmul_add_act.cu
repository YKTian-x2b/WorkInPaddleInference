#include <mutex>
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination_leaky_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/util/device_memory.h"
#include "paddle/phi/kernels/fusion/cutlass/gemm_epilogue/gemm_epilogue_util.h"

namespace phi{
namespace fusion{
namespace cutlass_internal{

cutlass::Status matmul_add_sm80_fp16_0(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_1(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_2(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_3(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_4(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_5(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_6(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_7(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_8(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_9(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_10(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_11(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_12(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_13(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_14(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_15(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_16(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_17(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_18(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_19(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_20(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_21(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_22(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_23(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_24(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_25(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_26(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_27(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_28(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_29(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_30(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_31(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_32(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_33(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_34(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_35(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_36(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_37(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_38(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_39(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_40(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_41(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_42(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_43(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_44(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_45(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_46(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_47(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_48(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_49(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_50(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_51(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_52(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_53(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_54(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_55(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_56(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_57(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_58(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_59(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_60(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_61(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_62(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_63(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_64(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_65(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_66(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_67(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_68(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_69(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_70(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_71(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_72(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_73(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_74(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_75(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_76(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_77(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_78(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_79(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_80(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_81(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_82(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_83(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_84(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_85(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_86(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_87(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_88(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_89(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_90(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_91(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_92(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_93(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_94(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_95(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_96(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_97(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_98(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_99(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_100(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_101(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_102(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_103(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_104(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_105(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_106(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_107(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_108(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_109(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_110(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_111(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_112(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_113(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_114(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_115(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_116(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_117(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_118(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_119(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_120(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_121(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_122(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_123(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_124(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_125(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_126(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_127(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_128(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_129(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_130(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_131(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_132(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_133(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_134(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_135(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_136(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_137(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_138(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_139(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_140(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_141(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_142(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_143(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_144(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_145(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_146(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_147(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_148(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_149(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_150(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_151(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_152(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_153(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_154(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_155(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_156(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_157(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_158(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_159(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_160(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_161(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_162(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_163(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_164(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_165(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_166(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_167(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_168(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_169(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_170(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_171(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_172(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_173(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_174(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_175(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_176(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_177(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_178(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_179(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_180(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_181(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_182(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_183(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_184(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_185(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_186(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_187(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_188(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_189(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_190(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_191(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_192(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_193(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_194(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_195(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_196(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_197(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_198(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_199(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_200(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_201(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_202(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_203(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_204(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_205(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_206(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_207(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_208(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_209(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_210(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_211(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_212(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_213(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_214(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_215(const GemmEpilogueAllParams& params);


std::vector<std::function<cutlass::Status(const GemmEpilogueAllParams)>>
    matmul_add_sm80_fp16_all_func =  {matmul_add_sm80_fp16_0, 
matmul_add_sm80_fp16_1, 
matmul_add_sm80_fp16_2, 
matmul_add_sm80_fp16_3, 
matmul_add_sm80_fp16_4, 
matmul_add_sm80_fp16_5, 
matmul_add_sm80_fp16_6, 
matmul_add_sm80_fp16_7, 
matmul_add_sm80_fp16_8, 
matmul_add_sm80_fp16_9, 
matmul_add_sm80_fp16_10, 
matmul_add_sm80_fp16_11, 
matmul_add_sm80_fp16_12, 
matmul_add_sm80_fp16_13, 
matmul_add_sm80_fp16_14, 
matmul_add_sm80_fp16_15, 
matmul_add_sm80_fp16_16, 
matmul_add_sm80_fp16_17, 
matmul_add_sm80_fp16_18, 
matmul_add_sm80_fp16_19, 
matmul_add_sm80_fp16_20, 
matmul_add_sm80_fp16_21, 
matmul_add_sm80_fp16_22, 
matmul_add_sm80_fp16_23, 
matmul_add_sm80_fp16_24, 
matmul_add_sm80_fp16_25, 
matmul_add_sm80_fp16_26, 
matmul_add_sm80_fp16_27, 
matmul_add_sm80_fp16_28, 
matmul_add_sm80_fp16_29, 
matmul_add_sm80_fp16_30, 
matmul_add_sm80_fp16_31, 
matmul_add_sm80_fp16_32, 
matmul_add_sm80_fp16_33, 
matmul_add_sm80_fp16_34, 
matmul_add_sm80_fp16_35, 
matmul_add_sm80_fp16_36, 
matmul_add_sm80_fp16_37, 
matmul_add_sm80_fp16_38, 
matmul_add_sm80_fp16_39, 
matmul_add_sm80_fp16_40, 
matmul_add_sm80_fp16_41, 
matmul_add_sm80_fp16_42, 
matmul_add_sm80_fp16_43, 
matmul_add_sm80_fp16_44, 
matmul_add_sm80_fp16_45, 
matmul_add_sm80_fp16_46, 
matmul_add_sm80_fp16_47, 
matmul_add_sm80_fp16_48, 
matmul_add_sm80_fp16_49, 
matmul_add_sm80_fp16_50, 
matmul_add_sm80_fp16_51, 
matmul_add_sm80_fp16_52, 
matmul_add_sm80_fp16_53, 
matmul_add_sm80_fp16_54, 
matmul_add_sm80_fp16_55, 
matmul_add_sm80_fp16_56, 
matmul_add_sm80_fp16_57, 
matmul_add_sm80_fp16_58, 
matmul_add_sm80_fp16_59, 
matmul_add_sm80_fp16_60, 
matmul_add_sm80_fp16_61, 
matmul_add_sm80_fp16_62, 
matmul_add_sm80_fp16_63, 
matmul_add_sm80_fp16_64, 
matmul_add_sm80_fp16_65, 
matmul_add_sm80_fp16_66, 
matmul_add_sm80_fp16_67, 
matmul_add_sm80_fp16_68, 
matmul_add_sm80_fp16_69, 
matmul_add_sm80_fp16_70, 
matmul_add_sm80_fp16_71, 
matmul_add_sm80_fp16_72, 
matmul_add_sm80_fp16_73, 
matmul_add_sm80_fp16_74, 
matmul_add_sm80_fp16_75, 
matmul_add_sm80_fp16_76, 
matmul_add_sm80_fp16_77, 
matmul_add_sm80_fp16_78, 
matmul_add_sm80_fp16_79, 
matmul_add_sm80_fp16_80, 
matmul_add_sm80_fp16_81, 
matmul_add_sm80_fp16_82, 
matmul_add_sm80_fp16_83, 
matmul_add_sm80_fp16_84, 
matmul_add_sm80_fp16_85, 
matmul_add_sm80_fp16_86, 
matmul_add_sm80_fp16_87, 
matmul_add_sm80_fp16_88, 
matmul_add_sm80_fp16_89, 
matmul_add_sm80_fp16_90, 
matmul_add_sm80_fp16_91, 
matmul_add_sm80_fp16_92, 
matmul_add_sm80_fp16_93, 
matmul_add_sm80_fp16_94, 
matmul_add_sm80_fp16_95, 
matmul_add_sm80_fp16_96, 
matmul_add_sm80_fp16_97, 
matmul_add_sm80_fp16_98, 
matmul_add_sm80_fp16_99, 
matmul_add_sm80_fp16_100, 
matmul_add_sm80_fp16_101, 
matmul_add_sm80_fp16_102, 
matmul_add_sm80_fp16_103, 
matmul_add_sm80_fp16_104, 
matmul_add_sm80_fp16_105, 
matmul_add_sm80_fp16_106, 
matmul_add_sm80_fp16_107, 
matmul_add_sm80_fp16_108, 
matmul_add_sm80_fp16_109, 
matmul_add_sm80_fp16_110, 
matmul_add_sm80_fp16_111, 
matmul_add_sm80_fp16_112, 
matmul_add_sm80_fp16_113, 
matmul_add_sm80_fp16_114, 
matmul_add_sm80_fp16_115, 
matmul_add_sm80_fp16_116, 
matmul_add_sm80_fp16_117, 
matmul_add_sm80_fp16_118, 
matmul_add_sm80_fp16_119, 
matmul_add_sm80_fp16_120, 
matmul_add_sm80_fp16_121, 
matmul_add_sm80_fp16_122, 
matmul_add_sm80_fp16_123, 
matmul_add_sm80_fp16_124, 
matmul_add_sm80_fp16_125, 
matmul_add_sm80_fp16_126, 
matmul_add_sm80_fp16_127, 
matmul_add_sm80_fp16_128, 
matmul_add_sm80_fp16_129, 
matmul_add_sm80_fp16_130, 
matmul_add_sm80_fp16_131, 
matmul_add_sm80_fp16_132, 
matmul_add_sm80_fp16_133, 
matmul_add_sm80_fp16_134, 
matmul_add_sm80_fp16_135, 
matmul_add_sm80_fp16_136, 
matmul_add_sm80_fp16_137, 
matmul_add_sm80_fp16_138, 
matmul_add_sm80_fp16_139, 
matmul_add_sm80_fp16_140, 
matmul_add_sm80_fp16_141, 
matmul_add_sm80_fp16_142, 
matmul_add_sm80_fp16_143, 
matmul_add_sm80_fp16_144, 
matmul_add_sm80_fp16_145, 
matmul_add_sm80_fp16_146, 
matmul_add_sm80_fp16_147, 
matmul_add_sm80_fp16_148, 
matmul_add_sm80_fp16_149, 
matmul_add_sm80_fp16_150, 
matmul_add_sm80_fp16_151, 
matmul_add_sm80_fp16_152, 
matmul_add_sm80_fp16_153, 
matmul_add_sm80_fp16_154, 
matmul_add_sm80_fp16_155, 
matmul_add_sm80_fp16_156, 
matmul_add_sm80_fp16_157, 
matmul_add_sm80_fp16_158, 
matmul_add_sm80_fp16_159, 
matmul_add_sm80_fp16_160, 
matmul_add_sm80_fp16_161, 
matmul_add_sm80_fp16_162, 
matmul_add_sm80_fp16_163, 
matmul_add_sm80_fp16_164, 
matmul_add_sm80_fp16_165, 
matmul_add_sm80_fp16_166, 
matmul_add_sm80_fp16_167, 
matmul_add_sm80_fp16_168, 
matmul_add_sm80_fp16_169, 
matmul_add_sm80_fp16_170, 
matmul_add_sm80_fp16_171, 
matmul_add_sm80_fp16_172, 
matmul_add_sm80_fp16_173, 
matmul_add_sm80_fp16_174, 
matmul_add_sm80_fp16_175, 
matmul_add_sm80_fp16_176, 
matmul_add_sm80_fp16_177, 
matmul_add_sm80_fp16_178, 
matmul_add_sm80_fp16_179, 
matmul_add_sm80_fp16_180, 
matmul_add_sm80_fp16_181, 
matmul_add_sm80_fp16_182, 
matmul_add_sm80_fp16_183, 
matmul_add_sm80_fp16_184, 
matmul_add_sm80_fp16_185, 
matmul_add_sm80_fp16_186, 
matmul_add_sm80_fp16_187, 
matmul_add_sm80_fp16_188, 
matmul_add_sm80_fp16_189, 
matmul_add_sm80_fp16_190, 
matmul_add_sm80_fp16_191, 
matmul_add_sm80_fp16_192, 
matmul_add_sm80_fp16_193, 
matmul_add_sm80_fp16_194, 
matmul_add_sm80_fp16_195, 
matmul_add_sm80_fp16_196, 
matmul_add_sm80_fp16_197, 
matmul_add_sm80_fp16_198, 
matmul_add_sm80_fp16_199, 
matmul_add_sm80_fp16_200, 
matmul_add_sm80_fp16_201, 
matmul_add_sm80_fp16_202, 
matmul_add_sm80_fp16_203, 
matmul_add_sm80_fp16_204, 
matmul_add_sm80_fp16_205, 
matmul_add_sm80_fp16_206, 
matmul_add_sm80_fp16_207, 
matmul_add_sm80_fp16_208, 
matmul_add_sm80_fp16_209, 
matmul_add_sm80_fp16_210, 
matmul_add_sm80_fp16_211, 
matmul_add_sm80_fp16_212, 
matmul_add_sm80_fp16_213, 
matmul_add_sm80_fp16_214, 
matmul_add_sm80_fp16_215, 
};

std::map<std::vector<int>, int> map_problem_matmul_add_sm80_fp16;
std::mutex matmul_add_sm80_fp16_mutex;

void matmul_add_sm80_fp16(GemmEpilogueAllParams params) {
  int m = params.m;
  int n = params.n;
  int k = params.k;
  int lda = params.lda;
  int ldb = params.ldb;
  int ldd = params.ldd;

  std::vector<int> problem_size = {m, n, k, lda, ldb, ldd};

  if (map_problem_matmul_add_sm80_fp16.count(problem_size)) {
    matmul_add_sm80_fp16_all_func[map_problem_matmul_add_sm80_fp16.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      matmul_add_sm80_fp16_all_func, params, MATMUL_ADD);

  std::lock_guard<std::mutex> guard(matmul_add_sm80_fp16_mutex);

  map_problem_matmul_add_sm80_fp16[problem_size] = best_config_index;
  matmul_add_sm80_fp16_all_func[best_config_index](params);
}

cutlass::Status matmul_add_relu_sm80_fp16_0(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_1(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_2(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_3(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_4(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_5(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_6(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_7(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_8(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_9(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_10(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_11(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_12(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_13(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_14(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_15(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_16(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_17(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_18(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_19(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_20(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_21(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_22(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_23(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_24(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_25(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_26(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_27(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_28(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_29(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_30(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_31(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_32(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_33(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_34(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_35(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_36(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_37(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_38(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_39(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_40(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_41(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_42(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_43(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_44(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_45(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_46(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_47(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_48(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_49(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_50(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_51(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_52(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_53(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_54(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_55(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_56(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_57(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_58(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_59(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_60(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_61(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_62(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_63(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_64(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_65(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_66(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_67(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_68(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_69(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_70(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_71(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_72(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_73(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_74(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_75(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_76(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_77(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_78(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_79(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_80(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_81(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_82(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_83(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_84(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_85(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_86(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_87(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_88(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_89(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_90(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_91(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_92(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_93(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_94(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_95(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_96(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_97(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_98(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_99(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_100(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_101(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_102(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_103(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_104(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_105(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_106(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_107(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_108(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_109(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_110(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_111(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_112(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_113(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_114(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_115(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_116(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_117(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_118(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_119(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_120(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_121(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_122(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_123(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_124(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_125(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_126(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_127(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_128(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_129(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_130(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_131(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_132(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_133(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_134(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_135(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_136(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_137(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_138(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_139(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_140(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_141(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_142(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_143(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_144(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_145(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_146(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_147(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_148(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_149(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_150(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_151(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_152(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_153(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_154(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_155(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_156(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_157(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_158(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_159(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_160(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_161(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_162(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_163(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_164(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_165(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_166(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_167(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_168(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_169(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_170(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_171(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_172(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_173(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_174(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_175(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_176(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_177(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_178(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_179(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_180(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_181(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_182(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_183(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_184(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_185(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_186(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_187(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_188(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_189(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_190(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_191(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_192(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_193(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_194(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_195(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_196(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_197(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_198(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_199(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_200(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_201(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_202(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_203(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_204(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_205(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_206(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_207(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_208(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_209(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_210(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_211(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_212(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_213(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_214(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_fp16_215(const GemmEpilogueAllParams& params);


std::vector<std::function<cutlass::Status(const GemmEpilogueAllParams)>>
    matmul_add_relu_sm80_fp16_all_func =  {matmul_add_relu_sm80_fp16_0, 
matmul_add_relu_sm80_fp16_1, 
matmul_add_relu_sm80_fp16_2, 
matmul_add_relu_sm80_fp16_3, 
matmul_add_relu_sm80_fp16_4, 
matmul_add_relu_sm80_fp16_5, 
matmul_add_relu_sm80_fp16_6, 
matmul_add_relu_sm80_fp16_7, 
matmul_add_relu_sm80_fp16_8, 
matmul_add_relu_sm80_fp16_9, 
matmul_add_relu_sm80_fp16_10, 
matmul_add_relu_sm80_fp16_11, 
matmul_add_relu_sm80_fp16_12, 
matmul_add_relu_sm80_fp16_13, 
matmul_add_relu_sm80_fp16_14, 
matmul_add_relu_sm80_fp16_15, 
matmul_add_relu_sm80_fp16_16, 
matmul_add_relu_sm80_fp16_17, 
matmul_add_relu_sm80_fp16_18, 
matmul_add_relu_sm80_fp16_19, 
matmul_add_relu_sm80_fp16_20, 
matmul_add_relu_sm80_fp16_21, 
matmul_add_relu_sm80_fp16_22, 
matmul_add_relu_sm80_fp16_23, 
matmul_add_relu_sm80_fp16_24, 
matmul_add_relu_sm80_fp16_25, 
matmul_add_relu_sm80_fp16_26, 
matmul_add_relu_sm80_fp16_27, 
matmul_add_relu_sm80_fp16_28, 
matmul_add_relu_sm80_fp16_29, 
matmul_add_relu_sm80_fp16_30, 
matmul_add_relu_sm80_fp16_31, 
matmul_add_relu_sm80_fp16_32, 
matmul_add_relu_sm80_fp16_33, 
matmul_add_relu_sm80_fp16_34, 
matmul_add_relu_sm80_fp16_35, 
matmul_add_relu_sm80_fp16_36, 
matmul_add_relu_sm80_fp16_37, 
matmul_add_relu_sm80_fp16_38, 
matmul_add_relu_sm80_fp16_39, 
matmul_add_relu_sm80_fp16_40, 
matmul_add_relu_sm80_fp16_41, 
matmul_add_relu_sm80_fp16_42, 
matmul_add_relu_sm80_fp16_43, 
matmul_add_relu_sm80_fp16_44, 
matmul_add_relu_sm80_fp16_45, 
matmul_add_relu_sm80_fp16_46, 
matmul_add_relu_sm80_fp16_47, 
matmul_add_relu_sm80_fp16_48, 
matmul_add_relu_sm80_fp16_49, 
matmul_add_relu_sm80_fp16_50, 
matmul_add_relu_sm80_fp16_51, 
matmul_add_relu_sm80_fp16_52, 
matmul_add_relu_sm80_fp16_53, 
matmul_add_relu_sm80_fp16_54, 
matmul_add_relu_sm80_fp16_55, 
matmul_add_relu_sm80_fp16_56, 
matmul_add_relu_sm80_fp16_57, 
matmul_add_relu_sm80_fp16_58, 
matmul_add_relu_sm80_fp16_59, 
matmul_add_relu_sm80_fp16_60, 
matmul_add_relu_sm80_fp16_61, 
matmul_add_relu_sm80_fp16_62, 
matmul_add_relu_sm80_fp16_63, 
matmul_add_relu_sm80_fp16_64, 
matmul_add_relu_sm80_fp16_65, 
matmul_add_relu_sm80_fp16_66, 
matmul_add_relu_sm80_fp16_67, 
matmul_add_relu_sm80_fp16_68, 
matmul_add_relu_sm80_fp16_69, 
matmul_add_relu_sm80_fp16_70, 
matmul_add_relu_sm80_fp16_71, 
matmul_add_relu_sm80_fp16_72, 
matmul_add_relu_sm80_fp16_73, 
matmul_add_relu_sm80_fp16_74, 
matmul_add_relu_sm80_fp16_75, 
matmul_add_relu_sm80_fp16_76, 
matmul_add_relu_sm80_fp16_77, 
matmul_add_relu_sm80_fp16_78, 
matmul_add_relu_sm80_fp16_79, 
matmul_add_relu_sm80_fp16_80, 
matmul_add_relu_sm80_fp16_81, 
matmul_add_relu_sm80_fp16_82, 
matmul_add_relu_sm80_fp16_83, 
matmul_add_relu_sm80_fp16_84, 
matmul_add_relu_sm80_fp16_85, 
matmul_add_relu_sm80_fp16_86, 
matmul_add_relu_sm80_fp16_87, 
matmul_add_relu_sm80_fp16_88, 
matmul_add_relu_sm80_fp16_89, 
matmul_add_relu_sm80_fp16_90, 
matmul_add_relu_sm80_fp16_91, 
matmul_add_relu_sm80_fp16_92, 
matmul_add_relu_sm80_fp16_93, 
matmul_add_relu_sm80_fp16_94, 
matmul_add_relu_sm80_fp16_95, 
matmul_add_relu_sm80_fp16_96, 
matmul_add_relu_sm80_fp16_97, 
matmul_add_relu_sm80_fp16_98, 
matmul_add_relu_sm80_fp16_99, 
matmul_add_relu_sm80_fp16_100, 
matmul_add_relu_sm80_fp16_101, 
matmul_add_relu_sm80_fp16_102, 
matmul_add_relu_sm80_fp16_103, 
matmul_add_relu_sm80_fp16_104, 
matmul_add_relu_sm80_fp16_105, 
matmul_add_relu_sm80_fp16_106, 
matmul_add_relu_sm80_fp16_107, 
matmul_add_relu_sm80_fp16_108, 
matmul_add_relu_sm80_fp16_109, 
matmul_add_relu_sm80_fp16_110, 
matmul_add_relu_sm80_fp16_111, 
matmul_add_relu_sm80_fp16_112, 
matmul_add_relu_sm80_fp16_113, 
matmul_add_relu_sm80_fp16_114, 
matmul_add_relu_sm80_fp16_115, 
matmul_add_relu_sm80_fp16_116, 
matmul_add_relu_sm80_fp16_117, 
matmul_add_relu_sm80_fp16_118, 
matmul_add_relu_sm80_fp16_119, 
matmul_add_relu_sm80_fp16_120, 
matmul_add_relu_sm80_fp16_121, 
matmul_add_relu_sm80_fp16_122, 
matmul_add_relu_sm80_fp16_123, 
matmul_add_relu_sm80_fp16_124, 
matmul_add_relu_sm80_fp16_125, 
matmul_add_relu_sm80_fp16_126, 
matmul_add_relu_sm80_fp16_127, 
matmul_add_relu_sm80_fp16_128, 
matmul_add_relu_sm80_fp16_129, 
matmul_add_relu_sm80_fp16_130, 
matmul_add_relu_sm80_fp16_131, 
matmul_add_relu_sm80_fp16_132, 
matmul_add_relu_sm80_fp16_133, 
matmul_add_relu_sm80_fp16_134, 
matmul_add_relu_sm80_fp16_135, 
matmul_add_relu_sm80_fp16_136, 
matmul_add_relu_sm80_fp16_137, 
matmul_add_relu_sm80_fp16_138, 
matmul_add_relu_sm80_fp16_139, 
matmul_add_relu_sm80_fp16_140, 
matmul_add_relu_sm80_fp16_141, 
matmul_add_relu_sm80_fp16_142, 
matmul_add_relu_sm80_fp16_143, 
matmul_add_relu_sm80_fp16_144, 
matmul_add_relu_sm80_fp16_145, 
matmul_add_relu_sm80_fp16_146, 
matmul_add_relu_sm80_fp16_147, 
matmul_add_relu_sm80_fp16_148, 
matmul_add_relu_sm80_fp16_149, 
matmul_add_relu_sm80_fp16_150, 
matmul_add_relu_sm80_fp16_151, 
matmul_add_relu_sm80_fp16_152, 
matmul_add_relu_sm80_fp16_153, 
matmul_add_relu_sm80_fp16_154, 
matmul_add_relu_sm80_fp16_155, 
matmul_add_relu_sm80_fp16_156, 
matmul_add_relu_sm80_fp16_157, 
matmul_add_relu_sm80_fp16_158, 
matmul_add_relu_sm80_fp16_159, 
matmul_add_relu_sm80_fp16_160, 
matmul_add_relu_sm80_fp16_161, 
matmul_add_relu_sm80_fp16_162, 
matmul_add_relu_sm80_fp16_163, 
matmul_add_relu_sm80_fp16_164, 
matmul_add_relu_sm80_fp16_165, 
matmul_add_relu_sm80_fp16_166, 
matmul_add_relu_sm80_fp16_167, 
matmul_add_relu_sm80_fp16_168, 
matmul_add_relu_sm80_fp16_169, 
matmul_add_relu_sm80_fp16_170, 
matmul_add_relu_sm80_fp16_171, 
matmul_add_relu_sm80_fp16_172, 
matmul_add_relu_sm80_fp16_173, 
matmul_add_relu_sm80_fp16_174, 
matmul_add_relu_sm80_fp16_175, 
matmul_add_relu_sm80_fp16_176, 
matmul_add_relu_sm80_fp16_177, 
matmul_add_relu_sm80_fp16_178, 
matmul_add_relu_sm80_fp16_179, 
matmul_add_relu_sm80_fp16_180, 
matmul_add_relu_sm80_fp16_181, 
matmul_add_relu_sm80_fp16_182, 
matmul_add_relu_sm80_fp16_183, 
matmul_add_relu_sm80_fp16_184, 
matmul_add_relu_sm80_fp16_185, 
matmul_add_relu_sm80_fp16_186, 
matmul_add_relu_sm80_fp16_187, 
matmul_add_relu_sm80_fp16_188, 
matmul_add_relu_sm80_fp16_189, 
matmul_add_relu_sm80_fp16_190, 
matmul_add_relu_sm80_fp16_191, 
matmul_add_relu_sm80_fp16_192, 
matmul_add_relu_sm80_fp16_193, 
matmul_add_relu_sm80_fp16_194, 
matmul_add_relu_sm80_fp16_195, 
matmul_add_relu_sm80_fp16_196, 
matmul_add_relu_sm80_fp16_197, 
matmul_add_relu_sm80_fp16_198, 
matmul_add_relu_sm80_fp16_199, 
matmul_add_relu_sm80_fp16_200, 
matmul_add_relu_sm80_fp16_201, 
matmul_add_relu_sm80_fp16_202, 
matmul_add_relu_sm80_fp16_203, 
matmul_add_relu_sm80_fp16_204, 
matmul_add_relu_sm80_fp16_205, 
matmul_add_relu_sm80_fp16_206, 
matmul_add_relu_sm80_fp16_207, 
matmul_add_relu_sm80_fp16_208, 
matmul_add_relu_sm80_fp16_209, 
matmul_add_relu_sm80_fp16_210, 
matmul_add_relu_sm80_fp16_211, 
matmul_add_relu_sm80_fp16_212, 
matmul_add_relu_sm80_fp16_213, 
matmul_add_relu_sm80_fp16_214, 
matmul_add_relu_sm80_fp16_215, 
};

std::map<std::vector<int>, int> map_problem_matmul_add_relu_sm80_fp16;
std::mutex matmul_add_relu_sm80_fp16_mutex;

void matmul_add_relu_sm80_fp16(GemmEpilogueAllParams params) {
  int m = params.m;
  int n = params.n;
  int k = params.k;
  int lda = params.lda;
  int ldb = params.ldb;
  int ldd = params.ldd;

  std::vector<int> problem_size = {m, n, k, lda, ldb, ldd};

  if (map_problem_matmul_add_relu_sm80_fp16.count(problem_size)) {
    matmul_add_relu_sm80_fp16_all_func[map_problem_matmul_add_relu_sm80_fp16.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      matmul_add_relu_sm80_fp16_all_func, params, MATMUL_ADD_RELU);

  std::lock_guard<std::mutex> guard(matmul_add_relu_sm80_fp16_mutex);

  map_problem_matmul_add_relu_sm80_fp16[problem_size] = best_config_index;
  matmul_add_relu_sm80_fp16_all_func[best_config_index](params);
}

cutlass::Status matmul_add_gelu_sm80_fp16_0(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_1(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_2(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_3(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_4(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_5(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_6(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_7(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_8(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_9(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_10(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_11(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_12(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_13(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_14(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_15(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_16(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_17(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_18(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_19(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_20(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_21(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_22(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_23(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_24(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_25(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_26(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_27(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_28(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_29(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_30(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_31(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_32(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_33(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_34(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_35(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_36(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_37(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_38(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_39(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_40(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_41(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_42(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_43(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_44(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_45(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_46(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_47(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_48(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_49(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_50(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_51(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_52(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_53(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_54(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_55(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_56(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_57(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_58(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_59(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_60(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_61(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_62(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_63(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_64(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_65(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_66(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_67(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_68(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_69(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_70(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_71(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_72(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_73(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_74(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_75(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_76(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_77(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_78(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_79(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_80(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_81(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_82(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_83(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_84(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_85(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_86(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_87(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_88(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_89(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_90(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_91(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_92(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_93(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_94(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_95(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_96(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_97(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_98(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_99(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_100(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_101(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_102(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_103(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_104(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_105(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_106(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_107(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_108(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_109(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_110(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_111(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_112(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_113(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_114(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_115(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_116(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_117(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_118(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_119(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_120(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_121(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_122(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_123(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_124(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_125(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_126(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_127(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_128(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_129(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_130(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_131(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_132(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_133(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_134(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_135(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_136(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_137(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_138(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_139(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_140(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_141(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_142(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_143(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_144(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_145(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_146(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_147(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_148(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_149(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_150(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_151(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_152(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_153(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_154(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_155(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_156(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_157(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_158(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_159(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_160(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_161(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_162(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_163(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_164(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_165(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_166(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_167(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_168(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_169(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_170(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_171(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_172(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_173(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_174(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_175(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_176(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_177(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_178(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_179(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_180(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_181(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_182(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_183(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_184(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_185(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_186(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_187(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_188(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_189(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_190(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_191(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_192(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_193(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_194(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_195(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_196(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_197(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_198(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_199(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_200(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_201(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_202(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_203(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_204(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_205(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_206(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_207(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_208(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_209(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_210(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_211(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_212(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_213(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_214(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_fp16_215(const GemmEpilogueAllParams& params);


std::vector<std::function<cutlass::Status(const GemmEpilogueAllParams)>>
    matmul_add_gelu_sm80_fp16_all_func =  {matmul_add_gelu_sm80_fp16_0, 
matmul_add_gelu_sm80_fp16_1, 
matmul_add_gelu_sm80_fp16_2, 
matmul_add_gelu_sm80_fp16_3, 
matmul_add_gelu_sm80_fp16_4, 
matmul_add_gelu_sm80_fp16_5, 
matmul_add_gelu_sm80_fp16_6, 
matmul_add_gelu_sm80_fp16_7, 
matmul_add_gelu_sm80_fp16_8, 
matmul_add_gelu_sm80_fp16_9, 
matmul_add_gelu_sm80_fp16_10, 
matmul_add_gelu_sm80_fp16_11, 
matmul_add_gelu_sm80_fp16_12, 
matmul_add_gelu_sm80_fp16_13, 
matmul_add_gelu_sm80_fp16_14, 
matmul_add_gelu_sm80_fp16_15, 
matmul_add_gelu_sm80_fp16_16, 
matmul_add_gelu_sm80_fp16_17, 
matmul_add_gelu_sm80_fp16_18, 
matmul_add_gelu_sm80_fp16_19, 
matmul_add_gelu_sm80_fp16_20, 
matmul_add_gelu_sm80_fp16_21, 
matmul_add_gelu_sm80_fp16_22, 
matmul_add_gelu_sm80_fp16_23, 
matmul_add_gelu_sm80_fp16_24, 
matmul_add_gelu_sm80_fp16_25, 
matmul_add_gelu_sm80_fp16_26, 
matmul_add_gelu_sm80_fp16_27, 
matmul_add_gelu_sm80_fp16_28, 
matmul_add_gelu_sm80_fp16_29, 
matmul_add_gelu_sm80_fp16_30, 
matmul_add_gelu_sm80_fp16_31, 
matmul_add_gelu_sm80_fp16_32, 
matmul_add_gelu_sm80_fp16_33, 
matmul_add_gelu_sm80_fp16_34, 
matmul_add_gelu_sm80_fp16_35, 
matmul_add_gelu_sm80_fp16_36, 
matmul_add_gelu_sm80_fp16_37, 
matmul_add_gelu_sm80_fp16_38, 
matmul_add_gelu_sm80_fp16_39, 
matmul_add_gelu_sm80_fp16_40, 
matmul_add_gelu_sm80_fp16_41, 
matmul_add_gelu_sm80_fp16_42, 
matmul_add_gelu_sm80_fp16_43, 
matmul_add_gelu_sm80_fp16_44, 
matmul_add_gelu_sm80_fp16_45, 
matmul_add_gelu_sm80_fp16_46, 
matmul_add_gelu_sm80_fp16_47, 
matmul_add_gelu_sm80_fp16_48, 
matmul_add_gelu_sm80_fp16_49, 
matmul_add_gelu_sm80_fp16_50, 
matmul_add_gelu_sm80_fp16_51, 
matmul_add_gelu_sm80_fp16_52, 
matmul_add_gelu_sm80_fp16_53, 
matmul_add_gelu_sm80_fp16_54, 
matmul_add_gelu_sm80_fp16_55, 
matmul_add_gelu_sm80_fp16_56, 
matmul_add_gelu_sm80_fp16_57, 
matmul_add_gelu_sm80_fp16_58, 
matmul_add_gelu_sm80_fp16_59, 
matmul_add_gelu_sm80_fp16_60, 
matmul_add_gelu_sm80_fp16_61, 
matmul_add_gelu_sm80_fp16_62, 
matmul_add_gelu_sm80_fp16_63, 
matmul_add_gelu_sm80_fp16_64, 
matmul_add_gelu_sm80_fp16_65, 
matmul_add_gelu_sm80_fp16_66, 
matmul_add_gelu_sm80_fp16_67, 
matmul_add_gelu_sm80_fp16_68, 
matmul_add_gelu_sm80_fp16_69, 
matmul_add_gelu_sm80_fp16_70, 
matmul_add_gelu_sm80_fp16_71, 
matmul_add_gelu_sm80_fp16_72, 
matmul_add_gelu_sm80_fp16_73, 
matmul_add_gelu_sm80_fp16_74, 
matmul_add_gelu_sm80_fp16_75, 
matmul_add_gelu_sm80_fp16_76, 
matmul_add_gelu_sm80_fp16_77, 
matmul_add_gelu_sm80_fp16_78, 
matmul_add_gelu_sm80_fp16_79, 
matmul_add_gelu_sm80_fp16_80, 
matmul_add_gelu_sm80_fp16_81, 
matmul_add_gelu_sm80_fp16_82, 
matmul_add_gelu_sm80_fp16_83, 
matmul_add_gelu_sm80_fp16_84, 
matmul_add_gelu_sm80_fp16_85, 
matmul_add_gelu_sm80_fp16_86, 
matmul_add_gelu_sm80_fp16_87, 
matmul_add_gelu_sm80_fp16_88, 
matmul_add_gelu_sm80_fp16_89, 
matmul_add_gelu_sm80_fp16_90, 
matmul_add_gelu_sm80_fp16_91, 
matmul_add_gelu_sm80_fp16_92, 
matmul_add_gelu_sm80_fp16_93, 
matmul_add_gelu_sm80_fp16_94, 
matmul_add_gelu_sm80_fp16_95, 
matmul_add_gelu_sm80_fp16_96, 
matmul_add_gelu_sm80_fp16_97, 
matmul_add_gelu_sm80_fp16_98, 
matmul_add_gelu_sm80_fp16_99, 
matmul_add_gelu_sm80_fp16_100, 
matmul_add_gelu_sm80_fp16_101, 
matmul_add_gelu_sm80_fp16_102, 
matmul_add_gelu_sm80_fp16_103, 
matmul_add_gelu_sm80_fp16_104, 
matmul_add_gelu_sm80_fp16_105, 
matmul_add_gelu_sm80_fp16_106, 
matmul_add_gelu_sm80_fp16_107, 
matmul_add_gelu_sm80_fp16_108, 
matmul_add_gelu_sm80_fp16_109, 
matmul_add_gelu_sm80_fp16_110, 
matmul_add_gelu_sm80_fp16_111, 
matmul_add_gelu_sm80_fp16_112, 
matmul_add_gelu_sm80_fp16_113, 
matmul_add_gelu_sm80_fp16_114, 
matmul_add_gelu_sm80_fp16_115, 
matmul_add_gelu_sm80_fp16_116, 
matmul_add_gelu_sm80_fp16_117, 
matmul_add_gelu_sm80_fp16_118, 
matmul_add_gelu_sm80_fp16_119, 
matmul_add_gelu_sm80_fp16_120, 
matmul_add_gelu_sm80_fp16_121, 
matmul_add_gelu_sm80_fp16_122, 
matmul_add_gelu_sm80_fp16_123, 
matmul_add_gelu_sm80_fp16_124, 
matmul_add_gelu_sm80_fp16_125, 
matmul_add_gelu_sm80_fp16_126, 
matmul_add_gelu_sm80_fp16_127, 
matmul_add_gelu_sm80_fp16_128, 
matmul_add_gelu_sm80_fp16_129, 
matmul_add_gelu_sm80_fp16_130, 
matmul_add_gelu_sm80_fp16_131, 
matmul_add_gelu_sm80_fp16_132, 
matmul_add_gelu_sm80_fp16_133, 
matmul_add_gelu_sm80_fp16_134, 
matmul_add_gelu_sm80_fp16_135, 
matmul_add_gelu_sm80_fp16_136, 
matmul_add_gelu_sm80_fp16_137, 
matmul_add_gelu_sm80_fp16_138, 
matmul_add_gelu_sm80_fp16_139, 
matmul_add_gelu_sm80_fp16_140, 
matmul_add_gelu_sm80_fp16_141, 
matmul_add_gelu_sm80_fp16_142, 
matmul_add_gelu_sm80_fp16_143, 
matmul_add_gelu_sm80_fp16_144, 
matmul_add_gelu_sm80_fp16_145, 
matmul_add_gelu_sm80_fp16_146, 
matmul_add_gelu_sm80_fp16_147, 
matmul_add_gelu_sm80_fp16_148, 
matmul_add_gelu_sm80_fp16_149, 
matmul_add_gelu_sm80_fp16_150, 
matmul_add_gelu_sm80_fp16_151, 
matmul_add_gelu_sm80_fp16_152, 
matmul_add_gelu_sm80_fp16_153, 
matmul_add_gelu_sm80_fp16_154, 
matmul_add_gelu_sm80_fp16_155, 
matmul_add_gelu_sm80_fp16_156, 
matmul_add_gelu_sm80_fp16_157, 
matmul_add_gelu_sm80_fp16_158, 
matmul_add_gelu_sm80_fp16_159, 
matmul_add_gelu_sm80_fp16_160, 
matmul_add_gelu_sm80_fp16_161, 
matmul_add_gelu_sm80_fp16_162, 
matmul_add_gelu_sm80_fp16_163, 
matmul_add_gelu_sm80_fp16_164, 
matmul_add_gelu_sm80_fp16_165, 
matmul_add_gelu_sm80_fp16_166, 
matmul_add_gelu_sm80_fp16_167, 
matmul_add_gelu_sm80_fp16_168, 
matmul_add_gelu_sm80_fp16_169, 
matmul_add_gelu_sm80_fp16_170, 
matmul_add_gelu_sm80_fp16_171, 
matmul_add_gelu_sm80_fp16_172, 
matmul_add_gelu_sm80_fp16_173, 
matmul_add_gelu_sm80_fp16_174, 
matmul_add_gelu_sm80_fp16_175, 
matmul_add_gelu_sm80_fp16_176, 
matmul_add_gelu_sm80_fp16_177, 
matmul_add_gelu_sm80_fp16_178, 
matmul_add_gelu_sm80_fp16_179, 
matmul_add_gelu_sm80_fp16_180, 
matmul_add_gelu_sm80_fp16_181, 
matmul_add_gelu_sm80_fp16_182, 
matmul_add_gelu_sm80_fp16_183, 
matmul_add_gelu_sm80_fp16_184, 
matmul_add_gelu_sm80_fp16_185, 
matmul_add_gelu_sm80_fp16_186, 
matmul_add_gelu_sm80_fp16_187, 
matmul_add_gelu_sm80_fp16_188, 
matmul_add_gelu_sm80_fp16_189, 
matmul_add_gelu_sm80_fp16_190, 
matmul_add_gelu_sm80_fp16_191, 
matmul_add_gelu_sm80_fp16_192, 
matmul_add_gelu_sm80_fp16_193, 
matmul_add_gelu_sm80_fp16_194, 
matmul_add_gelu_sm80_fp16_195, 
matmul_add_gelu_sm80_fp16_196, 
matmul_add_gelu_sm80_fp16_197, 
matmul_add_gelu_sm80_fp16_198, 
matmul_add_gelu_sm80_fp16_199, 
matmul_add_gelu_sm80_fp16_200, 
matmul_add_gelu_sm80_fp16_201, 
matmul_add_gelu_sm80_fp16_202, 
matmul_add_gelu_sm80_fp16_203, 
matmul_add_gelu_sm80_fp16_204, 
matmul_add_gelu_sm80_fp16_205, 
matmul_add_gelu_sm80_fp16_206, 
matmul_add_gelu_sm80_fp16_207, 
matmul_add_gelu_sm80_fp16_208, 
matmul_add_gelu_sm80_fp16_209, 
matmul_add_gelu_sm80_fp16_210, 
matmul_add_gelu_sm80_fp16_211, 
matmul_add_gelu_sm80_fp16_212, 
matmul_add_gelu_sm80_fp16_213, 
matmul_add_gelu_sm80_fp16_214, 
matmul_add_gelu_sm80_fp16_215, 
};

std::map<std::vector<int>, int> map_problem_matmul_add_gelu_sm80_fp16;
std::mutex matmul_add_gelu_sm80_fp16_mutex;

void matmul_add_gelu_sm80_fp16(GemmEpilogueAllParams params) {
  int m = params.m;
  int n = params.n;
  int k = params.k;
  int lda = params.lda;
  int ldb = params.ldb;
  int ldd = params.ldd;

  std::vector<int> problem_size = {m, n, k, lda, ldb, ldd};

  if (map_problem_matmul_add_gelu_sm80_fp16.count(problem_size)) {
    matmul_add_gelu_sm80_fp16_all_func[map_problem_matmul_add_gelu_sm80_fp16.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      matmul_add_gelu_sm80_fp16_all_func, params, MATMUL_ADD_GELU);

  std::lock_guard<std::mutex> guard(matmul_add_gelu_sm80_fp16_mutex);

  map_problem_matmul_add_gelu_sm80_fp16[problem_size] = best_config_index;
  matmul_add_gelu_sm80_fp16_all_func[best_config_index](params);
}

cutlass::Status matmul_add_sm80_bf16_0(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_1(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_2(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_3(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_4(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_5(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_6(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_7(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_8(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_9(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_10(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_11(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_12(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_13(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_14(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_15(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_16(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_17(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_18(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_19(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_20(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_21(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_22(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_23(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_24(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_25(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_26(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_27(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_28(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_29(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_30(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_31(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_32(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_33(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_34(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_35(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_36(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_37(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_38(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_39(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_40(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_41(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_42(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_43(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_44(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_45(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_46(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_47(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_48(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_49(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_50(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_51(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_52(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_53(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_54(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_55(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_56(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_57(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_58(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_59(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_60(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_61(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_62(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_63(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_64(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_65(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_66(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_67(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_68(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_69(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_70(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_71(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_72(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_73(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_74(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_75(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_76(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_77(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_78(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_79(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_80(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_81(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_82(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_83(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_84(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_85(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_86(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_87(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_88(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_89(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_90(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_91(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_92(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_93(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_94(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_95(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_96(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_97(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_98(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_99(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_100(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_101(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_102(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_103(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_104(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_105(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_106(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_107(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_108(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_109(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_110(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_111(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_112(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_113(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_114(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_115(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_116(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_117(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_118(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_119(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_120(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_121(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_122(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_123(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_124(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_125(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_126(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_127(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_128(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_129(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_130(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_131(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_132(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_133(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_134(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_135(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_136(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_137(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_138(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_139(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_140(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_141(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_142(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_143(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_144(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_145(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_146(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_147(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_148(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_149(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_150(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_151(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_152(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_153(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_154(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_155(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_156(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_157(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_158(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_159(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_160(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_161(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_162(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_163(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_164(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_165(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_166(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_167(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_168(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_169(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_170(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_171(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_172(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_173(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_174(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_175(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_176(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_177(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_178(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_179(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_180(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_181(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_182(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_183(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_184(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_185(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_186(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_187(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_188(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_189(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_190(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_191(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_192(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_193(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_194(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_195(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_196(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_197(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_198(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_199(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_200(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_201(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_202(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_203(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_204(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_205(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_206(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_207(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_208(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_209(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_210(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_211(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_212(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_213(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_214(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_bf16_215(const GemmEpilogueAllParams& params);


std::vector<std::function<cutlass::Status(const GemmEpilogueAllParams)>>
    matmul_add_sm80_bf16_all_func =  {matmul_add_sm80_bf16_0, 
matmul_add_sm80_bf16_1, 
matmul_add_sm80_bf16_2, 
matmul_add_sm80_bf16_3, 
matmul_add_sm80_bf16_4, 
matmul_add_sm80_bf16_5, 
matmul_add_sm80_bf16_6, 
matmul_add_sm80_bf16_7, 
matmul_add_sm80_bf16_8, 
matmul_add_sm80_bf16_9, 
matmul_add_sm80_bf16_10, 
matmul_add_sm80_bf16_11, 
matmul_add_sm80_bf16_12, 
matmul_add_sm80_bf16_13, 
matmul_add_sm80_bf16_14, 
matmul_add_sm80_bf16_15, 
matmul_add_sm80_bf16_16, 
matmul_add_sm80_bf16_17, 
matmul_add_sm80_bf16_18, 
matmul_add_sm80_bf16_19, 
matmul_add_sm80_bf16_20, 
matmul_add_sm80_bf16_21, 
matmul_add_sm80_bf16_22, 
matmul_add_sm80_bf16_23, 
matmul_add_sm80_bf16_24, 
matmul_add_sm80_bf16_25, 
matmul_add_sm80_bf16_26, 
matmul_add_sm80_bf16_27, 
matmul_add_sm80_bf16_28, 
matmul_add_sm80_bf16_29, 
matmul_add_sm80_bf16_30, 
matmul_add_sm80_bf16_31, 
matmul_add_sm80_bf16_32, 
matmul_add_sm80_bf16_33, 
matmul_add_sm80_bf16_34, 
matmul_add_sm80_bf16_35, 
matmul_add_sm80_bf16_36, 
matmul_add_sm80_bf16_37, 
matmul_add_sm80_bf16_38, 
matmul_add_sm80_bf16_39, 
matmul_add_sm80_bf16_40, 
matmul_add_sm80_bf16_41, 
matmul_add_sm80_bf16_42, 
matmul_add_sm80_bf16_43, 
matmul_add_sm80_bf16_44, 
matmul_add_sm80_bf16_45, 
matmul_add_sm80_bf16_46, 
matmul_add_sm80_bf16_47, 
matmul_add_sm80_bf16_48, 
matmul_add_sm80_bf16_49, 
matmul_add_sm80_bf16_50, 
matmul_add_sm80_bf16_51, 
matmul_add_sm80_bf16_52, 
matmul_add_sm80_bf16_53, 
matmul_add_sm80_bf16_54, 
matmul_add_sm80_bf16_55, 
matmul_add_sm80_bf16_56, 
matmul_add_sm80_bf16_57, 
matmul_add_sm80_bf16_58, 
matmul_add_sm80_bf16_59, 
matmul_add_sm80_bf16_60, 
matmul_add_sm80_bf16_61, 
matmul_add_sm80_bf16_62, 
matmul_add_sm80_bf16_63, 
matmul_add_sm80_bf16_64, 
matmul_add_sm80_bf16_65, 
matmul_add_sm80_bf16_66, 
matmul_add_sm80_bf16_67, 
matmul_add_sm80_bf16_68, 
matmul_add_sm80_bf16_69, 
matmul_add_sm80_bf16_70, 
matmul_add_sm80_bf16_71, 
matmul_add_sm80_bf16_72, 
matmul_add_sm80_bf16_73, 
matmul_add_sm80_bf16_74, 
matmul_add_sm80_bf16_75, 
matmul_add_sm80_bf16_76, 
matmul_add_sm80_bf16_77, 
matmul_add_sm80_bf16_78, 
matmul_add_sm80_bf16_79, 
matmul_add_sm80_bf16_80, 
matmul_add_sm80_bf16_81, 
matmul_add_sm80_bf16_82, 
matmul_add_sm80_bf16_83, 
matmul_add_sm80_bf16_84, 
matmul_add_sm80_bf16_85, 
matmul_add_sm80_bf16_86, 
matmul_add_sm80_bf16_87, 
matmul_add_sm80_bf16_88, 
matmul_add_sm80_bf16_89, 
matmul_add_sm80_bf16_90, 
matmul_add_sm80_bf16_91, 
matmul_add_sm80_bf16_92, 
matmul_add_sm80_bf16_93, 
matmul_add_sm80_bf16_94, 
matmul_add_sm80_bf16_95, 
matmul_add_sm80_bf16_96, 
matmul_add_sm80_bf16_97, 
matmul_add_sm80_bf16_98, 
matmul_add_sm80_bf16_99, 
matmul_add_sm80_bf16_100, 
matmul_add_sm80_bf16_101, 
matmul_add_sm80_bf16_102, 
matmul_add_sm80_bf16_103, 
matmul_add_sm80_bf16_104, 
matmul_add_sm80_bf16_105, 
matmul_add_sm80_bf16_106, 
matmul_add_sm80_bf16_107, 
matmul_add_sm80_bf16_108, 
matmul_add_sm80_bf16_109, 
matmul_add_sm80_bf16_110, 
matmul_add_sm80_bf16_111, 
matmul_add_sm80_bf16_112, 
matmul_add_sm80_bf16_113, 
matmul_add_sm80_bf16_114, 
matmul_add_sm80_bf16_115, 
matmul_add_sm80_bf16_116, 
matmul_add_sm80_bf16_117, 
matmul_add_sm80_bf16_118, 
matmul_add_sm80_bf16_119, 
matmul_add_sm80_bf16_120, 
matmul_add_sm80_bf16_121, 
matmul_add_sm80_bf16_122, 
matmul_add_sm80_bf16_123, 
matmul_add_sm80_bf16_124, 
matmul_add_sm80_bf16_125, 
matmul_add_sm80_bf16_126, 
matmul_add_sm80_bf16_127, 
matmul_add_sm80_bf16_128, 
matmul_add_sm80_bf16_129, 
matmul_add_sm80_bf16_130, 
matmul_add_sm80_bf16_131, 
matmul_add_sm80_bf16_132, 
matmul_add_sm80_bf16_133, 
matmul_add_sm80_bf16_134, 
matmul_add_sm80_bf16_135, 
matmul_add_sm80_bf16_136, 
matmul_add_sm80_bf16_137, 
matmul_add_sm80_bf16_138, 
matmul_add_sm80_bf16_139, 
matmul_add_sm80_bf16_140, 
matmul_add_sm80_bf16_141, 
matmul_add_sm80_bf16_142, 
matmul_add_sm80_bf16_143, 
matmul_add_sm80_bf16_144, 
matmul_add_sm80_bf16_145, 
matmul_add_sm80_bf16_146, 
matmul_add_sm80_bf16_147, 
matmul_add_sm80_bf16_148, 
matmul_add_sm80_bf16_149, 
matmul_add_sm80_bf16_150, 
matmul_add_sm80_bf16_151, 
matmul_add_sm80_bf16_152, 
matmul_add_sm80_bf16_153, 
matmul_add_sm80_bf16_154, 
matmul_add_sm80_bf16_155, 
matmul_add_sm80_bf16_156, 
matmul_add_sm80_bf16_157, 
matmul_add_sm80_bf16_158, 
matmul_add_sm80_bf16_159, 
matmul_add_sm80_bf16_160, 
matmul_add_sm80_bf16_161, 
matmul_add_sm80_bf16_162, 
matmul_add_sm80_bf16_163, 
matmul_add_sm80_bf16_164, 
matmul_add_sm80_bf16_165, 
matmul_add_sm80_bf16_166, 
matmul_add_sm80_bf16_167, 
matmul_add_sm80_bf16_168, 
matmul_add_sm80_bf16_169, 
matmul_add_sm80_bf16_170, 
matmul_add_sm80_bf16_171, 
matmul_add_sm80_bf16_172, 
matmul_add_sm80_bf16_173, 
matmul_add_sm80_bf16_174, 
matmul_add_sm80_bf16_175, 
matmul_add_sm80_bf16_176, 
matmul_add_sm80_bf16_177, 
matmul_add_sm80_bf16_178, 
matmul_add_sm80_bf16_179, 
matmul_add_sm80_bf16_180, 
matmul_add_sm80_bf16_181, 
matmul_add_sm80_bf16_182, 
matmul_add_sm80_bf16_183, 
matmul_add_sm80_bf16_184, 
matmul_add_sm80_bf16_185, 
matmul_add_sm80_bf16_186, 
matmul_add_sm80_bf16_187, 
matmul_add_sm80_bf16_188, 
matmul_add_sm80_bf16_189, 
matmul_add_sm80_bf16_190, 
matmul_add_sm80_bf16_191, 
matmul_add_sm80_bf16_192, 
matmul_add_sm80_bf16_193, 
matmul_add_sm80_bf16_194, 
matmul_add_sm80_bf16_195, 
matmul_add_sm80_bf16_196, 
matmul_add_sm80_bf16_197, 
matmul_add_sm80_bf16_198, 
matmul_add_sm80_bf16_199, 
matmul_add_sm80_bf16_200, 
matmul_add_sm80_bf16_201, 
matmul_add_sm80_bf16_202, 
matmul_add_sm80_bf16_203, 
matmul_add_sm80_bf16_204, 
matmul_add_sm80_bf16_205, 
matmul_add_sm80_bf16_206, 
matmul_add_sm80_bf16_207, 
matmul_add_sm80_bf16_208, 
matmul_add_sm80_bf16_209, 
matmul_add_sm80_bf16_210, 
matmul_add_sm80_bf16_211, 
matmul_add_sm80_bf16_212, 
matmul_add_sm80_bf16_213, 
matmul_add_sm80_bf16_214, 
matmul_add_sm80_bf16_215, 
};

std::map<std::vector<int>, int> map_problem_matmul_add_sm80_bf16;
std::mutex matmul_add_sm80_bf16_mutex;

void matmul_add_sm80_bf16(GemmEpilogueAllParams params) {
  int m = params.m;
  int n = params.n;
  int k = params.k;
  int lda = params.lda;
  int ldb = params.ldb;
  int ldd = params.ldd;

  std::vector<int> problem_size = {m, n, k, lda, ldb, ldd};

  if (map_problem_matmul_add_sm80_bf16.count(problem_size)) {
    matmul_add_sm80_bf16_all_func[map_problem_matmul_add_sm80_bf16.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      matmul_add_sm80_bf16_all_func, params, MATMUL_ADD);

  std::lock_guard<std::mutex> guard(matmul_add_sm80_bf16_mutex);

  map_problem_matmul_add_sm80_bf16[problem_size] = best_config_index;
  matmul_add_sm80_bf16_all_func[best_config_index](params);
}

cutlass::Status matmul_add_relu_sm80_bf16_0(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_1(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_2(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_3(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_4(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_5(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_6(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_7(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_8(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_9(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_10(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_11(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_12(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_13(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_14(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_15(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_16(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_17(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_18(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_19(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_20(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_21(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_22(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_23(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_24(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_25(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_26(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_27(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_28(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_29(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_30(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_31(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_32(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_33(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_34(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_35(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_36(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_37(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_38(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_39(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_40(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_41(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_42(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_43(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_44(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_45(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_46(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_47(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_48(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_49(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_50(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_51(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_52(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_53(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_54(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_55(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_56(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_57(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_58(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_59(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_60(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_61(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_62(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_63(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_64(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_65(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_66(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_67(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_68(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_69(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_70(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_71(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_72(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_73(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_74(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_75(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_76(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_77(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_78(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_79(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_80(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_81(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_82(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_83(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_84(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_85(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_86(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_87(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_88(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_89(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_90(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_91(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_92(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_93(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_94(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_95(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_96(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_97(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_98(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_99(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_100(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_101(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_102(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_103(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_104(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_105(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_106(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_107(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_108(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_109(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_110(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_111(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_112(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_113(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_114(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_115(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_116(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_117(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_118(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_119(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_120(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_121(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_122(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_123(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_124(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_125(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_126(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_127(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_128(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_129(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_130(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_131(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_132(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_133(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_134(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_135(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_136(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_137(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_138(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_139(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_140(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_141(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_142(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_143(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_144(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_145(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_146(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_147(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_148(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_149(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_150(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_151(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_152(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_153(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_154(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_155(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_156(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_157(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_158(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_159(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_160(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_161(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_162(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_163(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_164(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_165(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_166(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_167(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_168(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_169(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_170(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_171(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_172(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_173(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_174(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_175(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_176(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_177(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_178(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_179(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_180(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_181(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_182(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_183(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_184(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_185(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_186(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_187(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_188(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_189(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_190(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_191(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_192(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_193(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_194(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_195(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_196(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_197(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_198(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_199(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_200(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_201(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_202(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_203(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_204(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_205(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_206(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_207(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_208(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_209(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_210(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_211(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_212(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_213(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_214(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_relu_sm80_bf16_215(const GemmEpilogueAllParams& params);


std::vector<std::function<cutlass::Status(const GemmEpilogueAllParams)>>
    matmul_add_relu_sm80_bf16_all_func =  {matmul_add_relu_sm80_bf16_0, 
matmul_add_relu_sm80_bf16_1, 
matmul_add_relu_sm80_bf16_2, 
matmul_add_relu_sm80_bf16_3, 
matmul_add_relu_sm80_bf16_4, 
matmul_add_relu_sm80_bf16_5, 
matmul_add_relu_sm80_bf16_6, 
matmul_add_relu_sm80_bf16_7, 
matmul_add_relu_sm80_bf16_8, 
matmul_add_relu_sm80_bf16_9, 
matmul_add_relu_sm80_bf16_10, 
matmul_add_relu_sm80_bf16_11, 
matmul_add_relu_sm80_bf16_12, 
matmul_add_relu_sm80_bf16_13, 
matmul_add_relu_sm80_bf16_14, 
matmul_add_relu_sm80_bf16_15, 
matmul_add_relu_sm80_bf16_16, 
matmul_add_relu_sm80_bf16_17, 
matmul_add_relu_sm80_bf16_18, 
matmul_add_relu_sm80_bf16_19, 
matmul_add_relu_sm80_bf16_20, 
matmul_add_relu_sm80_bf16_21, 
matmul_add_relu_sm80_bf16_22, 
matmul_add_relu_sm80_bf16_23, 
matmul_add_relu_sm80_bf16_24, 
matmul_add_relu_sm80_bf16_25, 
matmul_add_relu_sm80_bf16_26, 
matmul_add_relu_sm80_bf16_27, 
matmul_add_relu_sm80_bf16_28, 
matmul_add_relu_sm80_bf16_29, 
matmul_add_relu_sm80_bf16_30, 
matmul_add_relu_sm80_bf16_31, 
matmul_add_relu_sm80_bf16_32, 
matmul_add_relu_sm80_bf16_33, 
matmul_add_relu_sm80_bf16_34, 
matmul_add_relu_sm80_bf16_35, 
matmul_add_relu_sm80_bf16_36, 
matmul_add_relu_sm80_bf16_37, 
matmul_add_relu_sm80_bf16_38, 
matmul_add_relu_sm80_bf16_39, 
matmul_add_relu_sm80_bf16_40, 
matmul_add_relu_sm80_bf16_41, 
matmul_add_relu_sm80_bf16_42, 
matmul_add_relu_sm80_bf16_43, 
matmul_add_relu_sm80_bf16_44, 
matmul_add_relu_sm80_bf16_45, 
matmul_add_relu_sm80_bf16_46, 
matmul_add_relu_sm80_bf16_47, 
matmul_add_relu_sm80_bf16_48, 
matmul_add_relu_sm80_bf16_49, 
matmul_add_relu_sm80_bf16_50, 
matmul_add_relu_sm80_bf16_51, 
matmul_add_relu_sm80_bf16_52, 
matmul_add_relu_sm80_bf16_53, 
matmul_add_relu_sm80_bf16_54, 
matmul_add_relu_sm80_bf16_55, 
matmul_add_relu_sm80_bf16_56, 
matmul_add_relu_sm80_bf16_57, 
matmul_add_relu_sm80_bf16_58, 
matmul_add_relu_sm80_bf16_59, 
matmul_add_relu_sm80_bf16_60, 
matmul_add_relu_sm80_bf16_61, 
matmul_add_relu_sm80_bf16_62, 
matmul_add_relu_sm80_bf16_63, 
matmul_add_relu_sm80_bf16_64, 
matmul_add_relu_sm80_bf16_65, 
matmul_add_relu_sm80_bf16_66, 
matmul_add_relu_sm80_bf16_67, 
matmul_add_relu_sm80_bf16_68, 
matmul_add_relu_sm80_bf16_69, 
matmul_add_relu_sm80_bf16_70, 
matmul_add_relu_sm80_bf16_71, 
matmul_add_relu_sm80_bf16_72, 
matmul_add_relu_sm80_bf16_73, 
matmul_add_relu_sm80_bf16_74, 
matmul_add_relu_sm80_bf16_75, 
matmul_add_relu_sm80_bf16_76, 
matmul_add_relu_sm80_bf16_77, 
matmul_add_relu_sm80_bf16_78, 
matmul_add_relu_sm80_bf16_79, 
matmul_add_relu_sm80_bf16_80, 
matmul_add_relu_sm80_bf16_81, 
matmul_add_relu_sm80_bf16_82, 
matmul_add_relu_sm80_bf16_83, 
matmul_add_relu_sm80_bf16_84, 
matmul_add_relu_sm80_bf16_85, 
matmul_add_relu_sm80_bf16_86, 
matmul_add_relu_sm80_bf16_87, 
matmul_add_relu_sm80_bf16_88, 
matmul_add_relu_sm80_bf16_89, 
matmul_add_relu_sm80_bf16_90, 
matmul_add_relu_sm80_bf16_91, 
matmul_add_relu_sm80_bf16_92, 
matmul_add_relu_sm80_bf16_93, 
matmul_add_relu_sm80_bf16_94, 
matmul_add_relu_sm80_bf16_95, 
matmul_add_relu_sm80_bf16_96, 
matmul_add_relu_sm80_bf16_97, 
matmul_add_relu_sm80_bf16_98, 
matmul_add_relu_sm80_bf16_99, 
matmul_add_relu_sm80_bf16_100, 
matmul_add_relu_sm80_bf16_101, 
matmul_add_relu_sm80_bf16_102, 
matmul_add_relu_sm80_bf16_103, 
matmul_add_relu_sm80_bf16_104, 
matmul_add_relu_sm80_bf16_105, 
matmul_add_relu_sm80_bf16_106, 
matmul_add_relu_sm80_bf16_107, 
matmul_add_relu_sm80_bf16_108, 
matmul_add_relu_sm80_bf16_109, 
matmul_add_relu_sm80_bf16_110, 
matmul_add_relu_sm80_bf16_111, 
matmul_add_relu_sm80_bf16_112, 
matmul_add_relu_sm80_bf16_113, 
matmul_add_relu_sm80_bf16_114, 
matmul_add_relu_sm80_bf16_115, 
matmul_add_relu_sm80_bf16_116, 
matmul_add_relu_sm80_bf16_117, 
matmul_add_relu_sm80_bf16_118, 
matmul_add_relu_sm80_bf16_119, 
matmul_add_relu_sm80_bf16_120, 
matmul_add_relu_sm80_bf16_121, 
matmul_add_relu_sm80_bf16_122, 
matmul_add_relu_sm80_bf16_123, 
matmul_add_relu_sm80_bf16_124, 
matmul_add_relu_sm80_bf16_125, 
matmul_add_relu_sm80_bf16_126, 
matmul_add_relu_sm80_bf16_127, 
matmul_add_relu_sm80_bf16_128, 
matmul_add_relu_sm80_bf16_129, 
matmul_add_relu_sm80_bf16_130, 
matmul_add_relu_sm80_bf16_131, 
matmul_add_relu_sm80_bf16_132, 
matmul_add_relu_sm80_bf16_133, 
matmul_add_relu_sm80_bf16_134, 
matmul_add_relu_sm80_bf16_135, 
matmul_add_relu_sm80_bf16_136, 
matmul_add_relu_sm80_bf16_137, 
matmul_add_relu_sm80_bf16_138, 
matmul_add_relu_sm80_bf16_139, 
matmul_add_relu_sm80_bf16_140, 
matmul_add_relu_sm80_bf16_141, 
matmul_add_relu_sm80_bf16_142, 
matmul_add_relu_sm80_bf16_143, 
matmul_add_relu_sm80_bf16_144, 
matmul_add_relu_sm80_bf16_145, 
matmul_add_relu_sm80_bf16_146, 
matmul_add_relu_sm80_bf16_147, 
matmul_add_relu_sm80_bf16_148, 
matmul_add_relu_sm80_bf16_149, 
matmul_add_relu_sm80_bf16_150, 
matmul_add_relu_sm80_bf16_151, 
matmul_add_relu_sm80_bf16_152, 
matmul_add_relu_sm80_bf16_153, 
matmul_add_relu_sm80_bf16_154, 
matmul_add_relu_sm80_bf16_155, 
matmul_add_relu_sm80_bf16_156, 
matmul_add_relu_sm80_bf16_157, 
matmul_add_relu_sm80_bf16_158, 
matmul_add_relu_sm80_bf16_159, 
matmul_add_relu_sm80_bf16_160, 
matmul_add_relu_sm80_bf16_161, 
matmul_add_relu_sm80_bf16_162, 
matmul_add_relu_sm80_bf16_163, 
matmul_add_relu_sm80_bf16_164, 
matmul_add_relu_sm80_bf16_165, 
matmul_add_relu_sm80_bf16_166, 
matmul_add_relu_sm80_bf16_167, 
matmul_add_relu_sm80_bf16_168, 
matmul_add_relu_sm80_bf16_169, 
matmul_add_relu_sm80_bf16_170, 
matmul_add_relu_sm80_bf16_171, 
matmul_add_relu_sm80_bf16_172, 
matmul_add_relu_sm80_bf16_173, 
matmul_add_relu_sm80_bf16_174, 
matmul_add_relu_sm80_bf16_175, 
matmul_add_relu_sm80_bf16_176, 
matmul_add_relu_sm80_bf16_177, 
matmul_add_relu_sm80_bf16_178, 
matmul_add_relu_sm80_bf16_179, 
matmul_add_relu_sm80_bf16_180, 
matmul_add_relu_sm80_bf16_181, 
matmul_add_relu_sm80_bf16_182, 
matmul_add_relu_sm80_bf16_183, 
matmul_add_relu_sm80_bf16_184, 
matmul_add_relu_sm80_bf16_185, 
matmul_add_relu_sm80_bf16_186, 
matmul_add_relu_sm80_bf16_187, 
matmul_add_relu_sm80_bf16_188, 
matmul_add_relu_sm80_bf16_189, 
matmul_add_relu_sm80_bf16_190, 
matmul_add_relu_sm80_bf16_191, 
matmul_add_relu_sm80_bf16_192, 
matmul_add_relu_sm80_bf16_193, 
matmul_add_relu_sm80_bf16_194, 
matmul_add_relu_sm80_bf16_195, 
matmul_add_relu_sm80_bf16_196, 
matmul_add_relu_sm80_bf16_197, 
matmul_add_relu_sm80_bf16_198, 
matmul_add_relu_sm80_bf16_199, 
matmul_add_relu_sm80_bf16_200, 
matmul_add_relu_sm80_bf16_201, 
matmul_add_relu_sm80_bf16_202, 
matmul_add_relu_sm80_bf16_203, 
matmul_add_relu_sm80_bf16_204, 
matmul_add_relu_sm80_bf16_205, 
matmul_add_relu_sm80_bf16_206, 
matmul_add_relu_sm80_bf16_207, 
matmul_add_relu_sm80_bf16_208, 
matmul_add_relu_sm80_bf16_209, 
matmul_add_relu_sm80_bf16_210, 
matmul_add_relu_sm80_bf16_211, 
matmul_add_relu_sm80_bf16_212, 
matmul_add_relu_sm80_bf16_213, 
matmul_add_relu_sm80_bf16_214, 
matmul_add_relu_sm80_bf16_215, 
};

std::map<std::vector<int>, int> map_problem_matmul_add_relu_sm80_bf16;
std::mutex matmul_add_relu_sm80_bf16_mutex;

void matmul_add_relu_sm80_bf16(GemmEpilogueAllParams params) {
  int m = params.m;
  int n = params.n;
  int k = params.k;
  int lda = params.lda;
  int ldb = params.ldb;
  int ldd = params.ldd;

  std::vector<int> problem_size = {m, n, k, lda, ldb, ldd};

  if (map_problem_matmul_add_relu_sm80_bf16.count(problem_size)) {
    matmul_add_relu_sm80_bf16_all_func[map_problem_matmul_add_relu_sm80_bf16.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      matmul_add_relu_sm80_bf16_all_func, params, MATMUL_ADD_RELU);

  std::lock_guard<std::mutex> guard(matmul_add_relu_sm80_bf16_mutex);

  map_problem_matmul_add_relu_sm80_bf16[problem_size] = best_config_index;
  matmul_add_relu_sm80_bf16_all_func[best_config_index](params);
}

cutlass::Status matmul_add_gelu_sm80_bf16_0(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_1(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_2(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_3(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_4(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_5(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_6(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_7(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_8(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_9(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_10(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_11(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_12(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_13(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_14(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_15(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_16(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_17(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_18(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_19(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_20(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_21(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_22(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_23(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_24(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_25(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_26(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_27(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_28(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_29(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_30(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_31(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_32(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_33(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_34(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_35(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_36(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_37(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_38(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_39(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_40(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_41(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_42(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_43(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_44(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_45(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_46(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_47(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_48(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_49(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_50(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_51(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_52(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_53(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_54(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_55(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_56(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_57(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_58(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_59(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_60(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_61(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_62(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_63(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_64(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_65(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_66(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_67(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_68(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_69(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_70(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_71(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_72(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_73(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_74(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_75(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_76(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_77(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_78(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_79(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_80(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_81(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_82(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_83(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_84(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_85(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_86(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_87(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_88(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_89(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_90(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_91(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_92(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_93(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_94(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_95(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_96(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_97(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_98(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_99(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_100(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_101(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_102(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_103(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_104(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_105(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_106(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_107(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_108(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_109(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_110(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_111(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_112(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_113(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_114(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_115(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_116(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_117(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_118(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_119(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_120(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_121(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_122(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_123(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_124(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_125(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_126(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_127(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_128(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_129(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_130(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_131(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_132(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_133(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_134(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_135(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_136(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_137(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_138(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_139(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_140(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_141(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_142(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_143(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_144(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_145(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_146(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_147(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_148(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_149(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_150(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_151(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_152(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_153(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_154(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_155(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_156(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_157(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_158(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_159(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_160(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_161(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_162(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_163(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_164(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_165(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_166(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_167(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_168(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_169(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_170(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_171(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_172(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_173(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_174(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_175(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_176(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_177(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_178(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_179(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_180(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_181(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_182(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_183(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_184(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_185(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_186(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_187(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_188(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_189(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_190(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_191(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_192(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_193(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_194(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_195(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_196(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_197(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_198(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_199(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_200(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_201(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_202(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_203(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_204(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_205(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_206(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_207(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_208(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_209(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_210(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_211(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_212(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_213(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_214(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_gelu_sm80_bf16_215(const GemmEpilogueAllParams& params);


std::vector<std::function<cutlass::Status(const GemmEpilogueAllParams)>>
    matmul_add_gelu_sm80_bf16_all_func =  {matmul_add_gelu_sm80_bf16_0, 
matmul_add_gelu_sm80_bf16_1, 
matmul_add_gelu_sm80_bf16_2, 
matmul_add_gelu_sm80_bf16_3, 
matmul_add_gelu_sm80_bf16_4, 
matmul_add_gelu_sm80_bf16_5, 
matmul_add_gelu_sm80_bf16_6, 
matmul_add_gelu_sm80_bf16_7, 
matmul_add_gelu_sm80_bf16_8, 
matmul_add_gelu_sm80_bf16_9, 
matmul_add_gelu_sm80_bf16_10, 
matmul_add_gelu_sm80_bf16_11, 
matmul_add_gelu_sm80_bf16_12, 
matmul_add_gelu_sm80_bf16_13, 
matmul_add_gelu_sm80_bf16_14, 
matmul_add_gelu_sm80_bf16_15, 
matmul_add_gelu_sm80_bf16_16, 
matmul_add_gelu_sm80_bf16_17, 
matmul_add_gelu_sm80_bf16_18, 
matmul_add_gelu_sm80_bf16_19, 
matmul_add_gelu_sm80_bf16_20, 
matmul_add_gelu_sm80_bf16_21, 
matmul_add_gelu_sm80_bf16_22, 
matmul_add_gelu_sm80_bf16_23, 
matmul_add_gelu_sm80_bf16_24, 
matmul_add_gelu_sm80_bf16_25, 
matmul_add_gelu_sm80_bf16_26, 
matmul_add_gelu_sm80_bf16_27, 
matmul_add_gelu_sm80_bf16_28, 
matmul_add_gelu_sm80_bf16_29, 
matmul_add_gelu_sm80_bf16_30, 
matmul_add_gelu_sm80_bf16_31, 
matmul_add_gelu_sm80_bf16_32, 
matmul_add_gelu_sm80_bf16_33, 
matmul_add_gelu_sm80_bf16_34, 
matmul_add_gelu_sm80_bf16_35, 
matmul_add_gelu_sm80_bf16_36, 
matmul_add_gelu_sm80_bf16_37, 
matmul_add_gelu_sm80_bf16_38, 
matmul_add_gelu_sm80_bf16_39, 
matmul_add_gelu_sm80_bf16_40, 
matmul_add_gelu_sm80_bf16_41, 
matmul_add_gelu_sm80_bf16_42, 
matmul_add_gelu_sm80_bf16_43, 
matmul_add_gelu_sm80_bf16_44, 
matmul_add_gelu_sm80_bf16_45, 
matmul_add_gelu_sm80_bf16_46, 
matmul_add_gelu_sm80_bf16_47, 
matmul_add_gelu_sm80_bf16_48, 
matmul_add_gelu_sm80_bf16_49, 
matmul_add_gelu_sm80_bf16_50, 
matmul_add_gelu_sm80_bf16_51, 
matmul_add_gelu_sm80_bf16_52, 
matmul_add_gelu_sm80_bf16_53, 
matmul_add_gelu_sm80_bf16_54, 
matmul_add_gelu_sm80_bf16_55, 
matmul_add_gelu_sm80_bf16_56, 
matmul_add_gelu_sm80_bf16_57, 
matmul_add_gelu_sm80_bf16_58, 
matmul_add_gelu_sm80_bf16_59, 
matmul_add_gelu_sm80_bf16_60, 
matmul_add_gelu_sm80_bf16_61, 
matmul_add_gelu_sm80_bf16_62, 
matmul_add_gelu_sm80_bf16_63, 
matmul_add_gelu_sm80_bf16_64, 
matmul_add_gelu_sm80_bf16_65, 
matmul_add_gelu_sm80_bf16_66, 
matmul_add_gelu_sm80_bf16_67, 
matmul_add_gelu_sm80_bf16_68, 
matmul_add_gelu_sm80_bf16_69, 
matmul_add_gelu_sm80_bf16_70, 
matmul_add_gelu_sm80_bf16_71, 
matmul_add_gelu_sm80_bf16_72, 
matmul_add_gelu_sm80_bf16_73, 
matmul_add_gelu_sm80_bf16_74, 
matmul_add_gelu_sm80_bf16_75, 
matmul_add_gelu_sm80_bf16_76, 
matmul_add_gelu_sm80_bf16_77, 
matmul_add_gelu_sm80_bf16_78, 
matmul_add_gelu_sm80_bf16_79, 
matmul_add_gelu_sm80_bf16_80, 
matmul_add_gelu_sm80_bf16_81, 
matmul_add_gelu_sm80_bf16_82, 
matmul_add_gelu_sm80_bf16_83, 
matmul_add_gelu_sm80_bf16_84, 
matmul_add_gelu_sm80_bf16_85, 
matmul_add_gelu_sm80_bf16_86, 
matmul_add_gelu_sm80_bf16_87, 
matmul_add_gelu_sm80_bf16_88, 
matmul_add_gelu_sm80_bf16_89, 
matmul_add_gelu_sm80_bf16_90, 
matmul_add_gelu_sm80_bf16_91, 
matmul_add_gelu_sm80_bf16_92, 
matmul_add_gelu_sm80_bf16_93, 
matmul_add_gelu_sm80_bf16_94, 
matmul_add_gelu_sm80_bf16_95, 
matmul_add_gelu_sm80_bf16_96, 
matmul_add_gelu_sm80_bf16_97, 
matmul_add_gelu_sm80_bf16_98, 
matmul_add_gelu_sm80_bf16_99, 
matmul_add_gelu_sm80_bf16_100, 
matmul_add_gelu_sm80_bf16_101, 
matmul_add_gelu_sm80_bf16_102, 
matmul_add_gelu_sm80_bf16_103, 
matmul_add_gelu_sm80_bf16_104, 
matmul_add_gelu_sm80_bf16_105, 
matmul_add_gelu_sm80_bf16_106, 
matmul_add_gelu_sm80_bf16_107, 
matmul_add_gelu_sm80_bf16_108, 
matmul_add_gelu_sm80_bf16_109, 
matmul_add_gelu_sm80_bf16_110, 
matmul_add_gelu_sm80_bf16_111, 
matmul_add_gelu_sm80_bf16_112, 
matmul_add_gelu_sm80_bf16_113, 
matmul_add_gelu_sm80_bf16_114, 
matmul_add_gelu_sm80_bf16_115, 
matmul_add_gelu_sm80_bf16_116, 
matmul_add_gelu_sm80_bf16_117, 
matmul_add_gelu_sm80_bf16_118, 
matmul_add_gelu_sm80_bf16_119, 
matmul_add_gelu_sm80_bf16_120, 
matmul_add_gelu_sm80_bf16_121, 
matmul_add_gelu_sm80_bf16_122, 
matmul_add_gelu_sm80_bf16_123, 
matmul_add_gelu_sm80_bf16_124, 
matmul_add_gelu_sm80_bf16_125, 
matmul_add_gelu_sm80_bf16_126, 
matmul_add_gelu_sm80_bf16_127, 
matmul_add_gelu_sm80_bf16_128, 
matmul_add_gelu_sm80_bf16_129, 
matmul_add_gelu_sm80_bf16_130, 
matmul_add_gelu_sm80_bf16_131, 
matmul_add_gelu_sm80_bf16_132, 
matmul_add_gelu_sm80_bf16_133, 
matmul_add_gelu_sm80_bf16_134, 
matmul_add_gelu_sm80_bf16_135, 
matmul_add_gelu_sm80_bf16_136, 
matmul_add_gelu_sm80_bf16_137, 
matmul_add_gelu_sm80_bf16_138, 
matmul_add_gelu_sm80_bf16_139, 
matmul_add_gelu_sm80_bf16_140, 
matmul_add_gelu_sm80_bf16_141, 
matmul_add_gelu_sm80_bf16_142, 
matmul_add_gelu_sm80_bf16_143, 
matmul_add_gelu_sm80_bf16_144, 
matmul_add_gelu_sm80_bf16_145, 
matmul_add_gelu_sm80_bf16_146, 
matmul_add_gelu_sm80_bf16_147, 
matmul_add_gelu_sm80_bf16_148, 
matmul_add_gelu_sm80_bf16_149, 
matmul_add_gelu_sm80_bf16_150, 
matmul_add_gelu_sm80_bf16_151, 
matmul_add_gelu_sm80_bf16_152, 
matmul_add_gelu_sm80_bf16_153, 
matmul_add_gelu_sm80_bf16_154, 
matmul_add_gelu_sm80_bf16_155, 
matmul_add_gelu_sm80_bf16_156, 
matmul_add_gelu_sm80_bf16_157, 
matmul_add_gelu_sm80_bf16_158, 
matmul_add_gelu_sm80_bf16_159, 
matmul_add_gelu_sm80_bf16_160, 
matmul_add_gelu_sm80_bf16_161, 
matmul_add_gelu_sm80_bf16_162, 
matmul_add_gelu_sm80_bf16_163, 
matmul_add_gelu_sm80_bf16_164, 
matmul_add_gelu_sm80_bf16_165, 
matmul_add_gelu_sm80_bf16_166, 
matmul_add_gelu_sm80_bf16_167, 
matmul_add_gelu_sm80_bf16_168, 
matmul_add_gelu_sm80_bf16_169, 
matmul_add_gelu_sm80_bf16_170, 
matmul_add_gelu_sm80_bf16_171, 
matmul_add_gelu_sm80_bf16_172, 
matmul_add_gelu_sm80_bf16_173, 
matmul_add_gelu_sm80_bf16_174, 
matmul_add_gelu_sm80_bf16_175, 
matmul_add_gelu_sm80_bf16_176, 
matmul_add_gelu_sm80_bf16_177, 
matmul_add_gelu_sm80_bf16_178, 
matmul_add_gelu_sm80_bf16_179, 
matmul_add_gelu_sm80_bf16_180, 
matmul_add_gelu_sm80_bf16_181, 
matmul_add_gelu_sm80_bf16_182, 
matmul_add_gelu_sm80_bf16_183, 
matmul_add_gelu_sm80_bf16_184, 
matmul_add_gelu_sm80_bf16_185, 
matmul_add_gelu_sm80_bf16_186, 
matmul_add_gelu_sm80_bf16_187, 
matmul_add_gelu_sm80_bf16_188, 
matmul_add_gelu_sm80_bf16_189, 
matmul_add_gelu_sm80_bf16_190, 
matmul_add_gelu_sm80_bf16_191, 
matmul_add_gelu_sm80_bf16_192, 
matmul_add_gelu_sm80_bf16_193, 
matmul_add_gelu_sm80_bf16_194, 
matmul_add_gelu_sm80_bf16_195, 
matmul_add_gelu_sm80_bf16_196, 
matmul_add_gelu_sm80_bf16_197, 
matmul_add_gelu_sm80_bf16_198, 
matmul_add_gelu_sm80_bf16_199, 
matmul_add_gelu_sm80_bf16_200, 
matmul_add_gelu_sm80_bf16_201, 
matmul_add_gelu_sm80_bf16_202, 
matmul_add_gelu_sm80_bf16_203, 
matmul_add_gelu_sm80_bf16_204, 
matmul_add_gelu_sm80_bf16_205, 
matmul_add_gelu_sm80_bf16_206, 
matmul_add_gelu_sm80_bf16_207, 
matmul_add_gelu_sm80_bf16_208, 
matmul_add_gelu_sm80_bf16_209, 
matmul_add_gelu_sm80_bf16_210, 
matmul_add_gelu_sm80_bf16_211, 
matmul_add_gelu_sm80_bf16_212, 
matmul_add_gelu_sm80_bf16_213, 
matmul_add_gelu_sm80_bf16_214, 
matmul_add_gelu_sm80_bf16_215, 
};

std::map<std::vector<int>, int> map_problem_matmul_add_gelu_sm80_bf16;
std::mutex matmul_add_gelu_sm80_bf16_mutex;

void matmul_add_gelu_sm80_bf16(GemmEpilogueAllParams params) {
  int m = params.m;
  int n = params.n;
  int k = params.k;
  int lda = params.lda;
  int ldb = params.ldb;
  int ldd = params.ldd;

  std::vector<int> problem_size = {m, n, k, lda, ldb, ldd};

  if (map_problem_matmul_add_gelu_sm80_bf16.count(problem_size)) {
    matmul_add_gelu_sm80_bf16_all_func[map_problem_matmul_add_gelu_sm80_bf16.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      matmul_add_gelu_sm80_bf16_all_func, params, MATMUL_ADD_GELU);

  std::lock_guard<std::mutex> guard(matmul_add_gelu_sm80_bf16_mutex);

  map_problem_matmul_add_gelu_sm80_bf16[problem_size] = best_config_index;
  matmul_add_gelu_sm80_bf16_all_func[best_config_index](params);
}

void MatmulAdd(GemmEpilogueAllParams params) {
    
    if (params.sm_version == 80 && params.data_type == GemmEpilogueDataType::fp16)
    {
        matmul_add_sm80_fp16(params);
    }

    if (params.sm_version == 80 && params.data_type == GemmEpilogueDataType::bf16)
    {
        matmul_add_sm80_bf16(params);
    }

}

void MatmulAddRelu(GemmEpilogueAllParams params) {
    
    if (params.sm_version == 80 && params.data_type == GemmEpilogueDataType::fp16)
    {
        matmul_add_relu_sm80_fp16(params);
    }

    if (params.sm_version == 80 && params.data_type == GemmEpilogueDataType::bf16)
    {
        matmul_add_relu_sm80_bf16(params);
    }

}

void MatmulAddGelu(GemmEpilogueAllParams params) {
    
    if (params.sm_version == 80 && params.data_type == GemmEpilogueDataType::fp16)
    {
        matmul_add_gelu_sm80_fp16(params);
    }

    if (params.sm_version == 80 && params.data_type == GemmEpilogueDataType::bf16)
    {
        matmul_add_gelu_sm80_bf16(params);
    }

}


}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
