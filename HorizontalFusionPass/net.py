import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from paddle import nn
import paddle

class myNet(nn.Layer):
    def __init__(self, bsz, seq_len, num_head, head_dim, num_layers):
        super().__init__()
        self.bsz = bsz
        self.seq_len = seq_len
        self.num_head = num_head
        self.head_dim = head_dim
        self.dim = num_head * head_dim
        self.num_layers = num_layers

        self.fc1_list = nn.LayerList([nn.Linear(self.dim, self.dim) for i in range(self.num_layers)])
        self.fc2_list = nn.LayerList([nn.Linear(self.dim, self.dim, bias_attr=False) for i in range(self.num_layers)])
        self.act = nn.ReLU() 
        self.custom_bias = paddle.create_parameter([self.dim], dtype='float32', is_bias=True)
        self.custom_ww1 = paddle.create_parameter([self.dim, self.dim], dtype='float32')
        self.custom_ww2 = paddle.create_parameter([self.dim, self.dim], dtype='float32')
        
    @paddle.incubate.jit.inference(enable_new_ir=True, 
                          cache_static_model=False,
                          exp_enable_use_cutlass=True,
                          delete_pass_lists=[]# "common_subexpression_elimination_pass", 
                          )
    def static_compute(self, x):
        out_left_list = []
        out_right_list = []
        x_left, x_right = paddle.split(x, 2, -1)
        for i in range(self.num_layers): 
            out_left_i = self.fc1_list[i](x_left) 
            out_left_i = self.act(out_left_i) 
            out_left_i = out_left_i.reshape([self.bsz, self.seq_len, self.num_head, self.head_dim])
            out_left_list.append(out_left_i)
            #
            out_right_i = self.fc2_list[i](x_right) 
            out_right_i = out_right_i.reshape([self.bsz, self.seq_len, self.num_head, self.head_dim])
            out_right_list.append(out_right_i)
        out_left_list.append(paddle.add(x_left, self.custom_bias))
        paddle.matmul(x_right, self.custom_ww1, transpose_x=False, transpose_y=False)
        paddle.matmul(x_right, self.custom_ww2, transpose_x=True, transpose_y=False)
        out_all = []
        out_all.extend(out_left_list)
        out_all.extend(out_right_list)
        return out_all
        
    def forward(self, x, x_ref):
        out_ref_left_list = []
        out_ref_right_list = []
        custom_bias_ref = paddle.zeros([self.dim])
        x_ref_left, x_ref_right = paddle.split(x_ref, 2, -1)
        for i in range(self.num_layers): 
            out_left_i = paddle.matmul(x_ref_left, self.fc1_list[i].weight)
            out_left_i = paddle.add(out_left_i, self.fc1_list[i].bias)
            out_left_i = paddle.nn.functional.relu(out_left_i)
            out_left_i = out_left_i.reshape([self.bsz, self.seq_len, self.num_head, self.head_dim])
            out_ref_left_list.append(out_left_i)
            #
            out_right_i = paddle.matmul(x_ref_right, self.fc2_list[i].weight)
            out_right_i = out_right_i.reshape([self.bsz, self.seq_len, self.num_head, self.head_dim])
            out_ref_right_list.append(out_right_i)
        out_ref_left_list.append(paddle.add(x_ref_left, custom_bias_ref))
        out_ref_all = []
        out_ref_all.extend(out_ref_left_list)
        out_ref_all.extend(out_ref_right_list)
        
        out_all = self.static_compute(x)
        return out_all, out_ref_all


if __name__ == "__main__":
    bsz = 2
    seq_len = 16
    num_head = 8
    head_dim = 32
    dtype_ = "float32"
    dim = num_head * head_dim
    num_layers = 3
    net = myNet(bsz, seq_len, num_head, head_dim, num_layers)
    x = paddle.randn([bsz, seq_len, 2 * dim], dtype=dtype_)
    x_ref = paddle.clone(x)
    out_all, out_ref_all = net(x, x_ref)

    assert len(out_all) == len(out_all)
    print("\n---------- valid ------------")
    for i in range(len(out_all)):
        print(f"maxdiff_left_{i}: ", paddle.max(paddle.abs(out_ref_all[i] - out_all[i]))) 
        
    for i in range(len(out_all)):
        print(f"all_close_left_{i} : ", paddle.allclose(out_ref_all[i], out_all[i], rtol=0., atol=1e-02).cpu().numpy())
