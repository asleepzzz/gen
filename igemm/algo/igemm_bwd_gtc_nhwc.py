################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020 Advanced Micro Devices, Inc.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# 
################################################################################
# pylint: disable=maybe-no-member


class igemm_bwd_gtc_nhwc_t(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        self.global_load_out = self.global_load_out_t(mc, self)
        self.global_load_wei = self.global_load_wei_t(mc, self)
        self.shared_store_out = self.shared_store_out_t(mc, self)
        self.shared_store_wei = self.shared_store_wei_t(mc, self)

        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        self.out_thread_copy_ndim = len(out_thread_copy_index)
        self.wei_thread_copy_ndim = len(wei_thread_copy_index)
        assert self.out_thread_copy_ndim in (1, 2)
        assert self.wei_thread_copy_ndim in (1, 2)

        ctrl_thread_mapping = ctrl_thread_mapping_t()
                #                        ->      MR x  NR x ML1 x NL1 x ML0 x NL0
        ctrl_thread_mapping.thread_lengths = [self.tunable.gemm_m_repeat, self.tunable.gemm_n_repeat, 1, 1, self.tunable.gemm_m_per_thread, self.tunable.gemm_n_per_thread]
        ctrl_thread_mapping.cluster_lengths = [1, 1, self.tunable.gemm_m_level1_cluster, self.tunable.gemm_n_level1_cluster, self.tunable.gemm_m_level0_cluster, self.tunable.gemm_n_level0_cluster]
        self.thread_mapping = igemm_thread_mapping_t(self.mc, ctrl_thread_mapping)

        self.coalescing_store_groups = igemm_next_pow2(self.tunable.coalescing_store_groups)
        ctrl_coalescing_store = ctrl_coalescing_store_t()
        ctrl_coalescing_store.ctm = ctrl_thread_mapping
        ctrl_coalescing_store.coalescing_groups = self.coalescing_store_groups
        ctrl_coalescing_store.data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        ctrl_coalescing_store.vector_write_out = 1                      # TODO: some cases this can be set to other value
        ctrl_coalescing_store.block_size = self.tunable.block_size

        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
        n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = self.get_dims_lengths()
        ctrl_coalescing_store.gemm_m_m0_m1 = [n_c0, n_c1]
        if gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C1_C0:
            ctrl_coalescing_store.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0


        ctrl_coalescing_store.adjust_optimal_coalescing_groups()        # in m1_m0 order, must adjust 
        self.coalescing_store = igemm_coalescing_store_t(mc, ctrl_coalescing_store)

        '''
         in generic tensor contraction, gemm_m direction always is *good* dimension, fwd:k0*k1, bwd:c0*c1, wrw:k0*k1
         hence we always want to split coalescing groups along m direction, to store c matrix
        '''
        assert (self.tunable.gemm_m_per_thread * self.tunable.gemm_m_repeat) % self.coalescing_store_groups == 0, \
            f"coalescing store groups should be divided by thread m {self.tunable.gemm_m_per_thread}x{self.tunable.gemm_m_repeat}"

        self.label_out = f"L_{self.name()}_out"
        self.dict_shifted_stride = dict()


        self.karg = self.kernel_karg_t(mc, self)
        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)

    def name(self):
        return igemm_gtc_encode_kernel_name(self.tunable)
    def try_shift_stride(self, gpr, shifter):
        assert type(gpr) is sym_t
        with self._deferred_context():
            if gpr.label not in self.dict_shifted_stride:
                self.dict_shifted_stride[gpr.label] = gpr
                self._emit(f"s_lshl_b32 s[{gpr()}], s[{gpr()}], {shifter}")
        return self._get_deferred()