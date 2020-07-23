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
from __future__ import print_function
import sys

from ..codegen import *
from .utility import *

class amdgpu_swap_sequencer_t(object):
    '''
    partial-transpose 2d matrix in register, by using swap.
    currently only consider continus register in same col, aka col major

    after transpose, the num of col/row should be the same

    And be aware that, this method still is not straight-forward and not optimal,
    for v_swap_b32 have half speed. In this case better use several tmp register serve as vector buffer
    Hopefully in the future could have full speed v_swap_b32

        k0 k1 k2 k3          k0 k1 k2 k3
    e0 0  2  4  6    =>  e0 0  1  2  3
    e1 1  3  5  7        e1 4  5  6  7

        k0 k1 k2 k3         k0 k1 k2 k3
    e0  0  4  8  c       e0 0  1  2  3
    e1  1  5  9  d   =>  e1 4  5  6  7
    e2  2  6  a  e       e2 8  9  a  b
    e3  3  7  b  f       e3 c  d  e  f
    '''
    def create_2d_swap(self):
        def init_2d_indice(row, col):
            indice_2d = []
            for r in range(row):
                indice_2d.append([r+c*row for c in range(col)])
            return indice_2d
        def check_row_can_omit_swap(indice_2d, cur_row):
            '''
            if current row already fit in vector pattern, can omit out
            '''
            row = len(indice_2d)
            col = len(indice_2d[0])
            targeting_vector_pattern = []
            for c in range(col):
                targeting_vector_pattern.append(c)
            vector_diff = []
            for c in range(col):
                vector_diff.append(abs(indice_2d[cur_row][c] - targeting_vector_pattern[c]))
            lasf_diff = vector_diff[0]
            #print('xxx {}'.format(vector_diff))
            if lasf_diff % 2 != 0:
                return False
            for c in range(1, col):
                if lasf_diff != vector_diff[c]:
                    return False
            return True
        def scan_2d_indice(indice_2d):
            def locate_indice(indice_2d, target_indice, start_row):
                row = len(indice_2d)
                col = len(indice_2d[0])
                (tr, tc) = (start_row, 0)
                found = False
                for tr in range(start_row, row):
                    for tc in range(0, col):
                        #print(target_indice, indice_2d[tr][tc])
                        if target_indice == indice_2d[tr][tc]:
                            found = True
                            break
                    if found:
                        break
                assert found
                return (tr, tc)
            swap_list = []
            row = len(indice_2d)
            col = len(indice_2d[0])

            class touch_row_t(object):
                def __init__(self, row):
                    self.row = row
                    self.row_touched = [ 0 for r in range(row)]
                    self.row_touched_index = 0
                def next_untouched_row(self):
                    for r in range(self.row_touched_index, self.row):
                        if self.row_touched[r] == 0:
                            self.row_touched_index = r
                            return r
                    assert False
                def touch(self, row_index):
                    self.row_touched[row_index] = 1
            touch_row = touch_row_t(row)
            for r in range(row):
                if check_row_can_omit_swap(indice_2d, r):
                    swap_list.append('unified for row {}'.format(r))
                    touch_row.touch( indice_2d[r][0] // col)
                    continue
                swap_list_per_row = []
                for c in range(col):
                    target_indice = touch_row.next_untouched_row()*col + c
                    origin_indice = indice_2d[r][c]
                    if origin_indice == target_indice:
                        continue
                    #print('to find:{}'.format(target_indice))
                    (tr, tc) = locate_indice(indice_2d, target_indice, r)
                    # swap and record indice
                    indice_2d[tr][tc] = origin_indice
                    indice_2d[r][c] = target_indice
                    #print('swapper:{}'.format(indice_2d))
                    swap_list_per_row.append((origin_indice, target_indice))
                swap_list.append(swap_list_per_row)
                touch_row.touch(r)
            return swap_list
        indice_2d = init_2d_indice(self.row, self.col)
        #print(indice_2d)
        swap_list = scan_2d_indice(indice_2d)
        return swap_list

    def __init__(self, row, col):
        assert col != 1 and row != 1
        self.col = col
        self.row = row
        self.swap_list = self.create_2d_swap()

    def __call__(self):
        '''
        return list of tuple of the row row_idx what swap should take
        '''
        return self.swap_list

class inst_ds_read2_likely_t(mc_base_t):
    '''
    generate ds_read2 if possible. otherwise fallback to ds_read
    Design this not as macro, but inlined into other LDS store operation
    So need upper caller to make sure the uniqueness
    '''
    def name(self):
        return ''

    def __init__(self, mc, vec_count, vec_byte, vec_stride, sld_base = 0):
        mc_base_t.__init__(self, mc)
        self.vec_count = vec_count
        self.vec_byte = vec_byte
        #assert vec_byte in (4, 8)
        self.vec_stride = vec_stride
        self.sld_base = sld_base
    
    def likely_read2_b32(self):
        if self.vec_byte != 4:
            return False
        if self.vec_count % 2 != 0:
            return False
        if (self.sld_base % 4 == 0) and (self.vec_stride % 4 == 0):
            if (self.sld_base // 4) + (self.vec_stride // 4) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_read2st64_b32(self):
        if self.vec_byte != 4:
            return False
        if self.vec_count % 2 != 0:
            return False
        if (self.sld_base % (4*64) == 0) and (self.vec_stride % 4 == 0):
            if (self.sld_base // (4*64)) + (self.vec_stride // (4*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_read2_b64(self):
        if self.vec_byte != 8:
            return False
        if self.vec_count % 2 != 0:
            return False
        if (self.sld_base % 8 == 0) and (self.vec_stride % 8 == 0):
            if (self.sld_base // 8) + (self.vec_stride // 8) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_read2st64_b64(self):
        if self.vec_byte != 8:
            return False
        if self.vec_count % 2 != 0:
            return False
        if (self.sld_base % (8*64) == 0) and (self.vec_stride % (8*64) == 0):
            if (self.sld_base // (8*64)) + (self.vec_stride // (8*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def __call__(self, v_dst, v_sld_os):
        v_dst = sym_t(v_dst)
        v_sld_os = sym_t(v_sld_os)
        def emit_read2_fallback():
            sldx1 = inst_ds_read_t(self.vec_byte)
            with self._deferred_context():
                for n in range(self.vec_count):
                    self._emit(sldx1(v_dst(n*(self.vec_byte // 4)), v_sld_os(), self.sld_base + n * self.vec_stride))
                #if self.vec_byte == 4:
                #    for n in range(self.vec_count):
                #        self._emit(f'ds_read_b32 v[{v_dst(n)}], v[{v_sld_os()}] offset:{self.sld_base + n * self.vec_stride}')
                #elif self.vec_byte == 8:
                #    for n in range(self.vec_count):
                #        self._emit(f'ds_read_b64 v[{v_dst((2*n, 2*n + 1))}], v[{v_sld_os()}] offset:{self.sld_base + n * self.vec_stride}')
#
                #elif self.vec_byte == 16:
                #    for n in range(self.vec_count):
                #        self._emit(f'ds_read_b128 v[{v_dst((4*n, 4*n + 3))}], v[{v_sld_os()}] offset:{self.sld_base + n * self.vec_stride}')
                #else:
                #    assert False, 'unsupported vector size'
            return self._get_deferred()

        def emit_read2_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2_b32 v[{v_dst((2*n, 2*n+1))}], v[{v_sld_os()}], offset0:{(self.sld_base//4)+2*n*(self.vec_stride//4)}, offset1:{(self.sld_base//4)+(2*n+1)*(self.vec_stride//4)}')
            return self._get_deferred()

        def emit_read2st64_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2st64_b32 v[{v_dst((2*n,2*n+1))}], v[{v_sld_os()}], offset0:{(self.sld_base//(4*64))+2*n*(self.vec_stride//(4*64))}, offset1:{(self.sld_base//(4*64))+(2*n+1)*(self.vec_stride//(4*64))}')
            return self._get_deferred()

        def emit_read2_b64():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2_b64 v[{v_dst((4*n, 4*n+3))}], v[{v_sld_os()}], offset0:{(self.sld_base//8)+2*n*(self.vec_stride//8)}, offset1:{(self.sld_base//8)+(2*n+1)*(self.vec_stride//8)}')
            return self._get_deferred()

        def emit_read2st64_b64():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2st64_b64 v[{v_dst((4*n,4*n+3))}], v[{v_sld_os()}], offset0:{(self.sld_base//(8*64))+2*n*(self.vec_stride//(8*64))}, offset1:{(self.sld_base//(8*64))+(2*n+1)*(self.vec_stride//(8*64))}')
            return self._get_deferred()

        def likely_emit():
            if self.vec_byte == 4:
                if self.likely_read2_b32():
                    return emit_read2_b32()
                if self.likely_read2st64_b32():
                    return emit_read2st64_b32()
                return emit_read2_fallback()
            if self.vec_byte == 8:
                if self.likely_read2_b64():
                    return emit_read2_b64()
                if self.likely_read2st64_b64():
                    return emit_read2st64_b64()
                return emit_read2_fallback()
            return emit_read2_fallback()

        return likely_emit()
    def emit(self):
        assert False, 'dont use emit of this'
    def get_issues(self):
        if self.vec_byte == 4:
            if self.likely_read2_b32() or self.likely_read2st64_b32():
                return self.vec_count // 2
        if self.vec_byte == 8:
            if self.likely_read2_b64() or self.likely_read2st64_b64():
                return self.vec_count // 2
        return self.vec_count
'''
class inst_ds_write2_likely_t(mc_base_t):   
    def name(self):
        return ''
    def __init__(self, mc, tunable, vec_count, vec_byte, vec_stride, sst_base):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.vec_count        = vec_count
        self.vec_byte     = vec_byte
        self.vec_stride   = vec_stride
        self.sst_base     = sst_base
    def likely_write2_b32(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % 4 == 0) and (self.vec_stride % 4 == 0):
            if (self.sst_base // 4) + (self.vec_stride // 4) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b32(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % (4*64) == 0) and (self.vec_stride % 4 == 0):
            if (self.sst_base // (4*64)) + (self.vec_stride // (4*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2_b64(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % 8 == 0) and (self.vec_stride % 8 == 0):
            if (self.sst_base // 8) + (self.vec_stride // 8) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b64(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % (8*64) == 0) and (self.vec_stride % (8*64) == 0):
            if (self.sst_base // (8*64)) + (self.vec_stride // (8*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def __call__(self, v_src, v_sst):
        v_src = sym_t(v_src)
        v_sst = sym_t(v_sst)
        def emit_write2_fallback():
            with self._deferred_context():
                if self.vec_byte == 1:
                    for n in range(self.vec_count):
                        self._emit('ds_write_b32 v[{}], v[{}] offset:{}'.format(v_sst(), v_src(n), self.sst_base + n * self.vec_stride))
                elif self.vec_byte == 2:
                    if self.vec_count == 1:
                        self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(), v_src(1), self.sst_base ))
                    else:
                        swap_start = (self.vec_count*self.vec_byte) // 2
                        for n in range(self.vec_count // 2):
                            self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n + 1), v_src(2*n + swap_start)))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(2*n), v_src(2*n + 1), self.sst_base + 2*n * self.vec_stride))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(2*n + swap_start) , v_src(2*n + swap_start + 1), self.sst_base + (2*n+1) * self.vec_stride))
                elif self.vec_byte == 4:
                    if self.vec_count == 1:
                        self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(), v_src(3), self.sst_base ))
                    else:
                        # though we use algorithm in swap_seq to interleave swap with ds_write, but it is still wise to use extra tmp register for swap is half speed
                        swap_list = amdgpu_swap_sequencer_t(self.vec_count , self.vec_byte)()
                        # print('self.vec_count:{}, self.vec_byte:{}, {}'.format(self.vec_count , self.vec_byte, swap_list))
                        for n in range(self.vec_count):
                            sw = swap_list[n]
                            if type(sw) is str:
                                pass
                            else:
                                for sw_item in sw:
                                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(sw_item[0]) , v_src(sw_item[1]) ))
                            self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(4*n), v_src(4*n + 3), self.sst_base + n * self.vec_stride))
                else:
                    assert False, 'unsupported vector size'
            return self._get_deferred()

        def emit_write2_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('ds_write2_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(v_sst(),
                                v_src(2*n), v_src(2*n+1),
                                (self.sst_base//4)+2*n*(self.vec_stride//4), (self.sst_base//4)+(2*n+1)*(self.vec_stride//4)))
            return self._get_deferred()

        def emit_write2st64_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('ds_write2st64_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(v_sst(),
                                v_src(2*n), v_src(2*n+1),
                                (self.sst_base//(4*64))+2*n*(self.vec_stride//(4*64)), (self.sst_base//(4*64))+(2*n+1)*(self.vec_stride//(4*64))))
            return self._get_deferred()

        def emit_write2_b64():
            swap_start = (self.vec_count*self.vec_byte) // 2
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n+1), v_src(2*n+swap_start)))
                    self._emit('ds_write2_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(v_sst(),
                            v_src(2*n), v_src(2*n+1), v_src(2*n+swap_start), v_src(2*n+swap_start+1),
                            (self.sst_base//8)+2*n*(self.vec_stride//8), (self.sst_base//8)+(2*n+1)*(self.vec_stride//8)))
            return self._get_deferred()

        def emit_write2st64_b64():
            swap_start = (self.vec_count*self.vec_byte) // 2
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n+1), v_src(2*n+swap_start)))
                    self._emit('ds_write2st64_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(v_sst(),
                            v_src(2*n), v_src(2*n+1), v_src(2*n+swap_start), v_src(2*n+swap_start+1),
                            (self.sst_base//(8*64))+2*n*(self.vec_stride//(8*64)), (self.sst_base//(8*64))+(2*n+1)*(self.vec_stride//(8*64))))
            return self._get_deferred()

        def likely_emit():
            if self.vec_byte == 1:
                if self.likely_write2_b32():
                    return emit_write2_b32()
                if self.likely_write2st64_b32():
                    return emit_write2st64_b32()
                return emit_write2_fallback()
            if self.vec_byte == 2:
                if self.likely_write2_b64():
                    return emit_write2_b64()
                if self.likely_write2st64_b64():
                    return emit_write2st64_b64()
                return emit_write2_fallback()
            return emit_write2_fallback()

        return likely_emit()
    def emit(self):
        assert False, 'dont use emit of this'
    def get_issues(self):
        if self.vec_byte == 1:
            if self.likely_write2_b32() or self.likely_write2st64_b32():
                return self.vec_count // 2
        if self.vec_byte == 2:
            if self.likely_write2_b64() or self.likely_write2st64_b64():
                return self.vec_count // 2
        return self.vec_count
'''

class inst_ds_write2_likely_t(mc_base_t):   
    def name(self):
        return ''
    def __init__(self, mc, vec_count, vec_byte, vec_stride, sst_base=0):
        mc_base_t.__init__(self, mc)
        self.vec_count    = vec_count
        self.vec_byte     = vec_byte
        self.vec_stride   = vec_stride
        self.sst_base     = sst_base
    def likely_write2_b32(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % 4 == 0) and (self.vec_stride % 4 == 0):
            if (self.sst_base // 4) + (self.vec_stride // 4) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b32(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % (4*64) == 0) and (self.vec_stride % 4 == 0):
            if (self.sst_base // (4*64)) + (self.vec_stride // (4*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2_b64(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % 8 == 0) and (self.vec_stride % 8 == 0):
            if (self.sst_base // 8) + (self.vec_stride // 8) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b64(self):
        if self.vec_count % 2 != 0:
            return False
        if (self.sst_base % (8*64) == 0) and (self.vec_stride % (8*64) == 0):
            if (self.sst_base // (8*64)) + (self.vec_stride // (8*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def __call__(self, v_sst, v_src):
        v_src = sym_t(v_src)
        v_sst = sym_t(v_sst)
        def emit_write2_fallback():
            sstx1 = inst_ds_write_t(self.vec_byte)
            with self._deferred_context():
                for n in range(self.vec_count):
                    self._emit(sstx1(v_sst(), v_src(n*(self.vec_byte // 4)), self.sst_base + n * self.vec_stride))
            return self._get_deferred()

        def emit_write2_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2_b32 v[{v_sst()}], v[{v_src(2*n)}], v[{v_src(2*n+1)}], offset0:{(self.sst_base//4)+2*n*(self.vec_stride//4)}, offset1:{(self.sst_base//4)+(2*n+1)*(self.vec_stride//4)}')
            return self._get_deferred()

        def emit_write2st64_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2st64_b32 v[{v_sst()}], v[{v_src(2*n)}], v[{v_src(2*n+1)}], offset0:{(self.sst_base//(4*64))+2*n*(self.vec_stride//(4*64))}, offset1:{(self.sst_base//(4*64))+(2*n+1)*(self.vec_stride//(4*64))}')
            return self._get_deferred()

        def emit_write2_b64():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2_b64 v[{v_sst()}], v[{v_src((4*n, 4*n+1))}], v[{v_src((4*n+2, 4*n+3))}], offset0:{(self.sst_base//8)+2*n*(self.vec_stride//8)}, offset1:{(self.sst_base//8)+(2*n+1)*(self.vec_stride//8)}')
            return self._get_deferred()

        def emit_write2st64_b64():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2st64_b64 v[{v_sst()}], v[{v_src((4*n, 4*n+1))}], v[{v_src((4*n+2, 4*n+3))}], offset0:{(self.sst_base//(8*64))+2*n*(self.vec_stride//(8*64))}, offset1:{(self.sst_base//(8*64))+(2*n+1)*(self.vec_stride//(8*64))}')
            return self._get_deferred()

        def likely_emit():
            if self.vec_byte == 4:
                if self.likely_write2_b32():
                    return emit_write2_b32()
                if self.likely_write2st64_b32():
                    return emit_write2st64_b32()
                return emit_write2_fallback()
            if self.vec_byte == 8:
                if self.likely_write2_b64():
                    return emit_write2_b64()
                if self.likely_write2st64_b64():
                    return emit_write2st64_b64()
                return emit_write2_fallback()
            return emit_write2_fallback()
        return likely_emit()

    def emit(self):
        assert False, 'dont use emit of this'
    def get_issues(self):
        if self.vec_byte == 4:
            if self.likely_write2_b32() or self.likely_write2st64_b32():
                return self.vec_count // 2
        if self.vec_byte == 8:
            if self.likely_write2_b64() or self.likely_write2st64_b64():
                return self.vec_count // 2
        return self.vec_count


class inst_ds_read_t(object):
    def __init__(self, bytes):
        self.bytes = bytes
    def get_offset(self, offset):
        return '' if offset == 0 else 'offset:{}'.format(offset)
    def __call__(self, vdst, vaddr, offset):
        if self.bytes == 4:
            return 'ds_read_b32 v[{}], v[{}] {}'.format(vdst, vaddr, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_read_b64 v[{}:{}+1], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_read_b96 v[{}:{}+2], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_read_b128 v[{}:{}+3], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        assert False
    def get_issues(self):
        return 1

class inst_ds_write_t(object):
    def __init__(self, bytes):
        self.bytes = bytes

    def get_offset(self, offset):
        if type(offset) is str:
            return 'offset:{}'.format(offset)
        if type(offset) is int:
            return '' if offset == 0 else 'offset:{}'.format(offset)
        assert False

    def __call__(self, vaddr, vdata, offset = 0):
        if self.bytes == 4:
            return 'ds_write_b32 v[{}], v[{}] {}'.format(vaddr, vdata, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_write_b64 v[{}], v[{}:{}+1] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_write_b96 v[{}], v[{}:{}+2] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_write_b128 v[{}], v[{}:{}+3] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        assert False

    def get_issues(self):
        return 1

class ctrl_2d_shared_store_t(object):
    '''
    d0xd1
    '''
    def __init__(self):
        self.length_d0 = 1        # is d0 is 1, it is indeed 1d access
        self.length_d1 = 1
        self.vector_d1 = 1
        # self.offset_d1 = 0      # base offset
        self.stride_d0 = 0      # stride
        self.precision = 'fp32'      # 'fp32', 'fp16', ...
        self.src_order = 0  # 0-d0,d1, 1-d1,d0

class macro_igemm_2d_shared_store_t(mc_base_t):
    def __init__(self, mc, ctrl):
        assert type(ctrl) is ctrl_2d_shared_store_t
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
    def name(self):
        ctrl = self.ctrl
        if ctrl.precision == "fp32":
            bits_str = 'b32'
        elif ctrl.precision in ("fp16", "bf16"):
            bits_str = 'b16'
        else:
            assert False

        if ctrl.vector_d1 == 4:
            vec_str = 'v4'
        elif ctrl.vector_d1 == 2:
            vec_str = 'v2'
        elif ctrl.vector_d1 == 1:
            vec_str = 'v1'
        else:
            assert False

        assert ctrl.length_d1 == ctrl.vector_d1

        return f".v_sst_so{ctrl.src_order}_{ctrl.length_d0}x{ctrl.length_d1}_{bits_str}_{vec_str}" + \
                ("" if ctrl.length_d0 == 1 else f"_st{ctrl.stride_d0}")

    def __call__(self, v_src, v_sst_os):
        return '{} {}, {}'.format(self.name(), v_src, v_sst_os)
    def emit(self):
        ctrl = self.ctrl
        assert ctrl.length_d1 == ctrl.vector_d1
        assert ctrl.precision == 'fp32', "TO BE supported"
        ds_write = inst_ds_write_t(ctrl.vector_d1 * 4)
        with self._emit_macro_indented('.macro {} v_src, v_sst_os'.format(self.name())):
            if ctrl.src_order == 0:
                for i_d0 in range(ctrl.length_d0):
                    self._emit(ds_write('\\v_sst_os', f'\\v_src+{i_d0*ctrl.vector_d1}', i_d0 * ctrl.stride_d0))
            else:
                assert "unimplemented"
    def get_issues(self):
        return self.ctrl.length_d0
