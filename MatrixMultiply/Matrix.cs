using System;
using System.Numerics;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace MatrixMultiply
{
    public class Matrix     // 行列クラス. 列優先で配置.
    {
        const float EPSILON = 1.0e-2f;
        static readonly ParallelOptions PARALLEL_OPTIONS;

        static Matrix()
        {
            PARALLEL_OPTIONS = new ParallelOptions();
            PARALLEL_OPTIONS.MaxDegreeOfParallelism = 8;
        }

        public int RowNum { get; }
        public int ColumnNum { get; }

        readonly float[] DATA;

        public float this[int i, int j]
        {
            get { return this.DATA[i + j * this.RowNum]; }
            set { this.DATA[i + j * this.RowNum] = value; }
        }

        public Matrix(int rowNum, int colNum)
        {
            this.RowNum = rowNum;
            this.ColumnNum = colNum;
            this.DATA = new float[this.RowNum * this.ColumnNum];
        }

        public Matrix(int rowNum, int colNum, params float[] elements)
        {
            if (elements.Length != rowNum * colNum)
                throw new ArgumentException("Invalid length.", nameof(elements));

            this.RowNum = rowNum;
            this.ColumnNum = colNum;
            this.DATA = Transpose(elements, this.ColumnNum, this.RowNum);
        }

        public void Init(float num = 0.0f)
        {
            for (var j = 0; j < this.ColumnNum; j++)
                for (var i = 0; i < this.RowNum; i++)
                    this[i, j] = num;
        }

        public override string ToString()
        {
            var elements = string.Empty;
            for (var i = 0; i < this.RowNum; i++)
            {
                for (var j = 0; j < this.ColumnNum; j++)
                {
                    elements += this[i, j];
                    if (j != this.ColumnNum - 1)
                        elements += ", ";
                }
                elements += "\n";
            }
            return $"{base.ToString()}\n{{\n{elements}}}";
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Matrix))
                return false;

            var matrix = obj as Matrix;
            if (matrix.RowNum != this.RowNum || matrix.ColumnNum != this.ColumnNum)
                return false;
            for (var i = 0; i < this.DATA.Length; i++)
                if ((matrix.DATA[i] - this.DATA[i]) > EPSILON)
                    return false;
            return true;
        }

        public static void Subtract(Matrix left, Matrix right, Matrix differ)
        {
            for (var j = 0; j < differ.ColumnNum; j++)
                for (var i = 0; i < differ.RowNum; i++)
                    differ[i, j] = left[i, j] - right[i, j];
        }

        public static void Mult_0(Matrix left, Matrix right, Matrix product)    // 通常実装
        {
            for (var i = 0; i < left.RowNum; i++)
                for (var j = 0; j < right.ColumnNum; j++)
                    for (var k = 0; k < right.RowNum; k++)
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static void ParallelMult_0(Matrix left, Matrix right, Matrix product)
        {
            Parallel.For(0, left.RowNum, (i) =>
            {
                for (var j = 0; j < right.ColumnNum; j++)
                    for (var k = 0; k < right.RowNum; k++)
                        product[i, j] += left[i, k] * right[k, j];
            });
        }

        public static void Mult_1(Matrix left, Matrix right, Matrix product)    // ループ交換法
        {
            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++) 
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static void ParallelMult_1(Matrix left, Matrix right, Matrix product)
        {
            Parallel.For(0, right.ColumnNum, (j) =>
            {
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
            });
        }

        public static void Mult_2(Matrix left, Matrix right, Matrix product)    // 4段ループアンローリング 
        {
            const int UNROLL_STEP_NUM = 4;

            var j_rest = right.ColumnNum % UNROLL_STEP_NUM;
            var k_rest = right.RowNum % UNROLL_STEP_NUM;

            for (var j = 0; j < right.ColumnNum - j_rest; j += UNROLL_STEP_NUM)
            {
                var j_1 = j + 1;
                var j_2 = j + 2;
                var j_3 = j + 3;
                for (var k = 0; k < right.RowNum - k_rest; k += UNROLL_STEP_NUM)
                {
                    var k_1 = k + 1;
                    var k_2 = k + 2;
                    var k_3 = k + 3;
                    var right_00 = right[k, j]; var right_01 = right[k_1, j]; var right_02 = right[k_2, j]; var right_03 = right[k_3, j];
                    var right_10 = right[k, j_1]; var right_11 = right[k_1, j_1]; var right_12 = right[k_2, j_1]; var right_13 = right[k_3, j_1];
                    var right_20 = right[k, j_2]; var right_21 = right[k_1, j_2]; var right_22 = right[k_2, j_2]; var right_23 = right[k_3, j_2];
                    var right_30 = right[k, j_3]; var right_31 = right[k_1, j_3]; var right_32 = right[k_2, j_3]; var right_33 = right[k_3, j_3];
                    for (var i = 0; i < left.RowNum; i++)
                    {
                        var left_0 = left[i, k];
                        var left_1 = left[i, k + 1];
                        var left_2 = left[i, k + 2];
                        var left_3 = left[i, k + 3];
                        product[i, j] += left_0 * right_00 + left_1 * right_01 + left_2 * right_02 + left_3 * right_03;
                        product[i, j_1] += left_0 * right_10 + left_1 * right_11 + left_2 * right_12 + left_3 * right_13;
                        product[i, j_2] += left_0 * right_20 + left_1 * right_21 + left_2 * right_22 + left_3 * right_23;
                        product[i, j_3] += left_0 * right_30 + left_1 * right_31 + left_2 * right_32 + left_3 * right_33;
                    }
                }
            }

            // 端数処理
            for (var j = right.ColumnNum - j_rest; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product.DATA[i + j * product.RowNum] += left.DATA[i + k * left.RowNum] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = right.RowNum - k_rest; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product.DATA[i + j * product.RowNum] += left.DATA[i + k * left.RowNum] * right[k, j];
        }

        public static void ParallelMult_2(Matrix left, Matrix right, Matrix product)
        {
            const int STEP_SIZE = 4;

            var j_blocks = right.ColumnNum / STEP_SIZE;
            var j_rest = right.ColumnNum % STEP_SIZE;
            var k_rest = right.RowNum % STEP_SIZE;

            Parallel.For(0, j_blocks, (block) =>
            {
                var j = block * STEP_SIZE;
                var j_1 = j + 1;
                var j_2 = j + 2;
                var j_3 = j + 3;
                for (var k = 0; k < right.RowNum - k_rest; k += STEP_SIZE)
                {
                    var k_1 = k + 1;
                    var k_2 = k + 2;
                    var k_3 = k + 3;
                    var right_00 = right[k, j]; var right_01 = right[k_1, j]; var right_02 = right[k_2, j]; var right_03 = right[k_3, j];
                    var right_10 = right[k, j_1]; var right_11 = right[k_1, j_1]; var right_12 = right[k_2, j_1]; var right_13 = right[k_3, j_1];
                    var right_20 = right[k, j_2]; var right_21 = right[k_1, j_2]; var right_22 = right[k_2, j_2]; var right_23 = right[k_3, j_2];
                    var right_30 = right[k, j_3]; var right_31 = right[k_1, j_3]; var right_32 = right[k_2, j_3]; var right_33 = right[k_3, j_3];
                    for (var i = 0; i < left.RowNum; i++)
                    {
                        var left_0 = left[i, k];
                        var left_1 = left[i, k + 1];
                        var left_2 = left[i, k + 2];
                        var left_3 = left[i, k + 3];
                        product[i, j] += left_0 * right_00 + left_1 * right_01 + left_2 * right_02 + left_3 * right_03;
                        product[i, j_1] += left_0 * right_10 + left_1 * right_11 + left_2 * right_12 + left_3 * right_13;
                        product[i, j_2] += left_0 * right_20 + left_1 * right_21 + left_2 * right_22 + left_3 * right_23;
                        product[i, j_3] += left_0 * right_30 + left_1 * right_31 + left_2 * right_32 + left_3 * right_33;
                    }
                }
            });

            // 端数処理
            for (var j = right.ColumnNum - j_rest; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = right.RowNum - k_rest; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static void Mult_3(Matrix left, Matrix right, Matrix product)    // キャッシュブロッキング + 4段ループアンローリング
        {
            const int L2_BLOCK_SIZE = 128;     
            const int L1_BLOCK_SIZE = 64;     
            const int UNROLL_STEP = 4;

            var j_rest = right.ColumnNum % L2_BLOCK_SIZE;
            var k_rest = right.RowNum % L2_BLOCK_SIZE;
            var i_rest = left.RowNum % L2_BLOCK_SIZE;
            for (var j = 0; j < right.ColumnNum - j_rest; j += L2_BLOCK_SIZE)
                for (var k = 0; k < right.RowNum - k_rest; k += L2_BLOCK_SIZE)
                    for (var i = 0; i < left.RowNum - i_rest; i += L2_BLOCK_SIZE)
                        for (var jj = j; jj < j + L2_BLOCK_SIZE; jj += L1_BLOCK_SIZE)
                            for (var kk = k; kk < k + L2_BLOCK_SIZE; kk += L1_BLOCK_SIZE)
                                for (var ii = i; ii < i + L2_BLOCK_SIZE; ii += L1_BLOCK_SIZE)
                                {
                                    for (var jjj = jj; jjj < jj + L1_BLOCK_SIZE; jjj += UNROLL_STEP)
                                    {
                                        var jjj_1 = jjj + 1;
                                        var jjj_2 = jjj + 2;
                                        var jjj_3 = jjj + 3;
                                        for (var kkk = kk; kkk < kk + L1_BLOCK_SIZE; kkk += UNROLL_STEP)
                                        {
                                            var kkk_1 = kkk + 1;
                                            var kkk_2 = kkk + 2;
                                            var kkk_3 = kkk + 3;
                                            var right_00 = right[kkk, jjj]; var right_01 = right[kkk_1, jjj]; var right_02 = right[kkk_2, jjj]; var right_03 = right[kkk_3, jjj];
                                            var right_10 = right[kkk, jjj_1]; var right_11 = right[kkk_1, jjj_1]; var right_12 = right[kkk_2, jjj_1]; var right_13 = right[kkk_3, jjj_1];
                                            var right_20 = right[kkk, jjj_2]; var right_21 = right[kkk_1, jjj_2]; var right_22 = right[kkk_2, jjj_2]; var right_23 = right[kkk_3, jjj_2];
                                            var right_30 = right[kkk, jjj_3]; var right_31 = right[kkk_1, jjj_3]; var right_32 = right[kkk_2, jjj_3]; var right_33 = right[kkk_3, jjj_3];
                                            for (var iii = ii; iii < ii + L1_BLOCK_SIZE; iii++)
                                            {
                                                var left_0 = left[iii, kkk];
                                                var left_1 = left[iii, kkk + 1];
                                                var left_2 = left[iii, kkk + 2];
                                                var left_3 = left[iii, kkk + 3]; 
                                                product[iii, jjj] += left_0 * right_00 + left_1 * right_01 + left_2 * right_02 + left_3 * right_03;
                                                product[iii, jjj_1] += left_0 * right_10 + left_1 * right_11 + left_2 * right_12 + left_3 * right_13;
                                                product[iii, jjj_2] += left_0 * right_20 + left_1 * right_21 + left_2 * right_22 + left_3 * right_23;
                                                product[iii, jjj_3] += left_0 * right_30 + left_1 * right_31 + left_2 * right_32 + left_3 * right_33;
                                            }
                                        }
                                    }
                                }

            // 端数処理
            for (var j = right.ColumnNum - j_rest; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = right.RowNum - k_rest; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = left.RowNum - i_rest; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static void ParallelMult_3(Matrix left, Matrix right, Matrix product)
        {
            const int L2_BLOCK_SIZE = 128;      
            const int L1_BLOCK_SIZE = 64;      
            const int UNROLL_STEP = 4;

            var jBlockNum = right.ColumnNum / L2_BLOCK_SIZE;
            var j_rest = right.ColumnNum % L2_BLOCK_SIZE;
            var k_rest = right.RowNum % L2_BLOCK_SIZE;
            var i_rest = left.RowNum % L2_BLOCK_SIZE;
            Parallel.For(0, jBlockNum, PARALLEL_OPTIONS, (blockId) =>
            {
                var j = blockId * L2_BLOCK_SIZE;
                for (var k = 0; k < right.RowNum - k_rest; k += L2_BLOCK_SIZE)
                    for (var i = 0; i < left.RowNum - i_rest; i += L2_BLOCK_SIZE)
                        for (var jj = j; jj < j + L2_BLOCK_SIZE; jj += L1_BLOCK_SIZE)
                            for (var kk = k; kk < k + L2_BLOCK_SIZE; kk += L1_BLOCK_SIZE)
                                for (var ii = i; ii < i + L2_BLOCK_SIZE; ii += L1_BLOCK_SIZE)
                                {
                                    for (var jjj = jj; jjj < jj + L1_BLOCK_SIZE; jjj += UNROLL_STEP)
                                    {
                                        var jjj_1 = jjj + 1;
                                        var jjj_2 = jjj + 2;
                                        var jjj_3 = jjj + 3;
                                        for (var kkk = kk; kkk < kk + L1_BLOCK_SIZE; kkk += UNROLL_STEP)
                                        {
                                            var kkk_1 = kkk + 1;
                                            var kkk_2 = kkk + 2;
                                            var kkk_3 = kkk + 3;
                                            for (var iii = ii; iii < ii + L1_BLOCK_SIZE; iii++)
                                            {
                                                var left_0 = left[iii, kkk];
                                                var left_1 = left[iii, kkk + 1];
                                                var left_2 = left[iii, kkk + 2];
                                                var left_3 = left[iii, kkk + 3];
                                                product[iii, jjj] += left_0 * right[kkk, jjj] + left_1 * right[kkk_1, jjj] + left_2 * right[kkk_2, jjj] + left_3 * right[kkk_3, jjj];
                                                product[iii, jjj_1] += left_0 * right[kkk, jjj_1] + left_1 * right[kkk_1, jjj_1] + left_2 * right[kkk_2, jjj_1] + left_3 * right[kkk_3, jjj_1];
                                                product[iii, jjj_2] += left_0 * right[kkk, jjj_2] + left_1 * right[kkk_1, jjj_2] + left_2 * right[kkk_2, jjj_2] + left_3 * right[kkk_3, jjj_2];
                                                product[iii, jjj_3] += left_0 * right[kkk, jjj_3] + left_1 * right[kkk_1, jjj_3] + left_2 * right[kkk_2, jjj_3] + left_3 * right[kkk_3, jjj_3];
                                            }
                                        }
                                    }
                                }
            });

            // 端数処理
            for (var j = right.ColumnNum - j_rest; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = right.RowNum - k_rest; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = left.RowNum - i_rest; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static unsafe void Mult_4(Matrix left, Matrix right, Matrix product)    // キャッシュブロッキング + AVX
        {
            const int L2_BLOCK_SIZE = 128;      
            const int L1_BLOCK_SIZE = 64;     
            const int AVX_FLOAT_VEC_DIM = 256 / (sizeof(float) * 8);

            var jBlockNum = right.ColumnNum / L2_BLOCK_SIZE;
            var j_rest = right.ColumnNum % L2_BLOCK_SIZE;
            var k_rest = right.RowNum % L2_BLOCK_SIZE;
            var i_rest = left.RowNum % L2_BLOCK_SIZE;
            for (var j = 0; j < right.ColumnNum - j_rest; j += L2_BLOCK_SIZE)
                for (var k = 0; k < right.RowNum - k_rest; k += L2_BLOCK_SIZE)
                    for (var i = 0; i < left.RowNum - i_rest; i += L2_BLOCK_SIZE)
                        for (var jj = j; jj < j + L2_BLOCK_SIZE; jj += L1_BLOCK_SIZE)
                            for (var kk = k; kk < k + L2_BLOCK_SIZE; kk += L1_BLOCK_SIZE)
                                for (var ii = i; ii < i + L2_BLOCK_SIZE; ii += L1_BLOCK_SIZE)
                                    for (var jjj = jj; jjj < jj + L1_BLOCK_SIZE; jjj++)
                                        for (var kkk = kk; kkk < kk + L1_BLOCK_SIZE; kkk++)
                                        {
                                            fixed (float* leftPtr = &left.DATA[ii + kkk * left.RowNum])
                                            fixed (float* productPtr = &product.DATA[ii + jjj * product.RowNum])
                                            {
                                                var rightReg = Vector256.Create(right[kkk, jjj]);
                                                var leftReg_0 = Avx.LoadVector256(&leftPtr[0]);
                                                var productReg_0 = Avx.LoadVector256(&productPtr[0]);
                                                productReg_0 = Fma.MultiplyAdd(leftReg_0, rightReg, productReg_0);

                                                var leftReg_1 = Avx.LoadVector256(&leftPtr[AVX_FLOAT_VEC_DIM]);
                                                var productReg_1 = Avx.LoadVector256(&productPtr[AVX_FLOAT_VEC_DIM]);
                                                productReg_1 = Fma.MultiplyAdd(leftReg_1, rightReg, productReg_1);

                                                var leftReg_2 = Avx.LoadVector256(&leftPtr[2 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_2 = Avx.LoadVector256(&productPtr[2 * AVX_FLOAT_VEC_DIM]);
                                                productReg_2 = Fma.MultiplyAdd(leftReg_2, rightReg, productReg_2);

                                                var leftReg_3 = Avx.LoadVector256(&leftPtr[3 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_3 = Avx.LoadVector256(&productPtr[3 * AVX_FLOAT_VEC_DIM]);
                                                productReg_3 = Fma.MultiplyAdd(leftReg_3, rightReg, productReg_3);

                                                var leftReg_4 = Avx.LoadVector256(&leftPtr[4 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_4 = Avx.LoadVector256(&productPtr[4 * AVX_FLOAT_VEC_DIM]);
                                                productReg_4 = Fma.MultiplyAdd(leftReg_4, rightReg, productReg_4);

                                                var leftReg_5 = Avx.LoadVector256(&leftPtr[5 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_5 = Avx.LoadVector256(&productPtr[5 * AVX_FLOAT_VEC_DIM]);
                                                productReg_5 = Fma.MultiplyAdd(leftReg_5, rightReg, productReg_5);

                                                var leftReg_6 = Avx.LoadVector256(&leftPtr[6 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_6 = Avx.LoadVector256(&productPtr[6 * AVX_FLOAT_VEC_DIM]);
                                                productReg_6 = Fma.MultiplyAdd(leftReg_6, rightReg, productReg_6);

                                                var leftReg_7 = Avx.LoadVector256(&leftPtr[7 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_7 = Avx.LoadVector256(&productPtr[7 * AVX_FLOAT_VEC_DIM]);
                                                productReg_7 = Fma.MultiplyAdd(leftReg_7, rightReg, productReg_7);

                                                Avx.Store(&productPtr[0], productReg_0);
                                                Avx.Store(&productPtr[AVX_FLOAT_VEC_DIM], productReg_1);
                                                Avx.Store(&productPtr[2 * AVX_FLOAT_VEC_DIM], productReg_2);
                                                Avx.Store(&productPtr[3 * AVX_FLOAT_VEC_DIM], productReg_3);
                                                Avx.Store(&productPtr[4 * AVX_FLOAT_VEC_DIM], productReg_4);
                                                Avx.Store(&productPtr[5 * AVX_FLOAT_VEC_DIM], productReg_5);
                                                Avx.Store(&productPtr[6 * AVX_FLOAT_VEC_DIM], productReg_6);
                                                Avx.Store(&productPtr[7 * AVX_FLOAT_VEC_DIM], productReg_7);
                                            }
                                        }

            // 端数処理
            for (var j = right.ColumnNum - j_rest; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = right.RowNum - k_rest; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = left.RowNum - i_rest; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static unsafe void ParallelMult_4(Matrix left, Matrix right, Matrix product)    
        {
            const int L2_BLOCK_SIZE = 128;      
            const int L1_BLOCK_SIZE = 64;       
            const int AVX_FLOAT_VEC_DIM = 256 / (sizeof(float) * 8);

            var jBlockNum = right.ColumnNum / L2_BLOCK_SIZE;
            var j_rest = right.ColumnNum % L2_BLOCK_SIZE;
            var k_rest = right.RowNum % L2_BLOCK_SIZE;
            var i_rest = left.RowNum % L2_BLOCK_SIZE;
            Parallel.For(0, jBlockNum, PARALLEL_OPTIONS, (blockId) =>
            {
                var j = blockId * L2_BLOCK_SIZE;
                for (var k = 0; k < right.RowNum - k_rest; k += L2_BLOCK_SIZE)
                    for (var i = 0; i < left.RowNum - i_rest; i += L2_BLOCK_SIZE)
                        for (var jj = j; jj < j + L2_BLOCK_SIZE; jj += L1_BLOCK_SIZE)
                            for (var kk = k; kk < k + L2_BLOCK_SIZE; kk += L1_BLOCK_SIZE)
                                for (var ii = i; ii < i + L2_BLOCK_SIZE; ii += L1_BLOCK_SIZE)
                                    for (var jjj = jj; jjj < jj + L1_BLOCK_SIZE; jjj++)
                                        for (var kkk = kk; kkk < kk + L1_BLOCK_SIZE; kkk++)
                                        {
                                            fixed (float* leftPtr = &left.DATA[ii + kkk * left.RowNum])
                                            fixed (float* productPtr = &product.DATA[ii + jjj * product.RowNum])
                                            {
                                                var rightReg = Vector256.Create(right[kkk, jjj]);
                                                var leftReg_0 = Avx.LoadVector256(&leftPtr[0]);
                                                var productReg_0 = Avx.LoadVector256(&productPtr[0]);
                                                productReg_0 = Fma.MultiplyAdd(leftReg_0, rightReg, productReg_0);

                                                var leftReg_1 = Avx.LoadVector256(&leftPtr[AVX_FLOAT_VEC_DIM]);
                                                var productReg_1 = Avx.LoadVector256(&productPtr[AVX_FLOAT_VEC_DIM]);
                                                productReg_1 = Fma.MultiplyAdd(leftReg_1, rightReg, productReg_1);

                                                var leftReg_2 = Avx.LoadVector256(&leftPtr[2 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_2 = Avx.LoadVector256(&productPtr[2 * AVX_FLOAT_VEC_DIM]);
                                                productReg_2 = Fma.MultiplyAdd(leftReg_2, rightReg, productReg_2);

                                                var leftReg_3 = Avx.LoadVector256(&leftPtr[3 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_3 = Avx.LoadVector256(&productPtr[3 * AVX_FLOAT_VEC_DIM]);
                                                productReg_3 = Fma.MultiplyAdd(leftReg_3, rightReg, productReg_3);

                                                var leftReg_4 = Avx.LoadVector256(&leftPtr[4 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_4 = Avx.LoadVector256(&productPtr[4 * AVX_FLOAT_VEC_DIM]);
                                                productReg_4 = Fma.MultiplyAdd(leftReg_4, rightReg, productReg_4);

                                                var leftReg_5 = Avx.LoadVector256(&leftPtr[5 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_5 = Avx.LoadVector256(&productPtr[5 * AVX_FLOAT_VEC_DIM]);
                                                productReg_5 = Fma.MultiplyAdd(leftReg_5, rightReg, productReg_5);

                                                var leftReg_6 = Avx.LoadVector256(&leftPtr[6 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_6 = Avx.LoadVector256(&productPtr[6 * AVX_FLOAT_VEC_DIM]);
                                                productReg_6 = Fma.MultiplyAdd(leftReg_6, rightReg, productReg_6);

                                                var leftReg_7 = Avx.LoadVector256(&leftPtr[7 * AVX_FLOAT_VEC_DIM]);
                                                var productReg_7 = Avx.LoadVector256(&productPtr[7 * AVX_FLOAT_VEC_DIM]);
                                                productReg_7 = Fma.MultiplyAdd(leftReg_7, rightReg, productReg_7);

                                                Avx.Store(&productPtr[0], productReg_0);
                                                Avx.Store(&productPtr[AVX_FLOAT_VEC_DIM], productReg_1);
                                                Avx.Store(&productPtr[2 * AVX_FLOAT_VEC_DIM], productReg_2);
                                                Avx.Store(&productPtr[3 * AVX_FLOAT_VEC_DIM], productReg_3);
                                                Avx.Store(&productPtr[4 * AVX_FLOAT_VEC_DIM], productReg_4);
                                                Avx.Store(&productPtr[5 * AVX_FLOAT_VEC_DIM], productReg_5);
                                                Avx.Store(&productPtr[6 * AVX_FLOAT_VEC_DIM], productReg_6);
                                                Avx.Store(&productPtr[7 * AVX_FLOAT_VEC_DIM], productReg_7);
                                            }
                                        }
            });

            // 端数処理
            for (var j = right.ColumnNum - j_rest; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = right.RowNum - k_rest; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = left.RowNum - i_rest; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static void Mult_5(Matrix left, Matrix right, Matrix product)    // キャッシュブロッキング + Matrix4x4
        {
            const int L2_BLOCK_SIZE = 128;
            const int L1_BLOCK_SIZE = 64;
            const int PART_MATRIX_DIM = 4;

            var j_rest = right.ColumnNum % L2_BLOCK_SIZE;
            var k_rest = right.RowNum % L2_BLOCK_SIZE;
            var i_rest = left.RowNum % L2_BLOCK_SIZE;
            for (var j = 0; j < right.ColumnNum - j_rest; j += L2_BLOCK_SIZE)
                for (var k = 0; k < right.RowNum - k_rest; k += L2_BLOCK_SIZE)
                    for (var i = 0; i < left.RowNum - i_rest; i += L2_BLOCK_SIZE)
                        for (var jj = j; jj < j + L2_BLOCK_SIZE; jj += L1_BLOCK_SIZE)
                            for (var kk = k; kk < k + L2_BLOCK_SIZE; kk += L1_BLOCK_SIZE)
                                for (var ii = i; ii < i + L2_BLOCK_SIZE; ii += L1_BLOCK_SIZE)
                                    for (var jjj = jj; jjj < jj + L1_BLOCK_SIZE; jjj += PART_MATRIX_DIM)
                                    {
                                        var jjj_1 = jjj + 1;
                                        var jjj_2 = jjj + 2;
                                        var jjj_3 = jjj + 3;
                                        for (var kkk = kk; kkk < kk + L1_BLOCK_SIZE; kkk += PART_MATRIX_DIM)
                                        {
                                            var kkk_1 = kkk + 1;
                                            var kkk_2 = kkk + 2;
                                            var kkk_3 = kkk + 3;

                                            var right4x4 = new Matrix4x4(right[kkk, jjj], right[kkk, jjj_1], right[kkk, jjj_2], right[kkk, jjj_3],
                                                                         right[kkk_1, jjj], right[kkk_1, jjj_1], right[kkk_1, jjj_2], right[kkk_1, jjj_3],
                                                                         right[kkk_2, jjj], right[kkk_2, jjj_1], right[kkk_2, jjj_2], right[kkk_2, jjj_3],
                                                                         right[kkk_3, jjj], right[kkk_3, jjj_1], right[kkk_3, jjj_2], right[kkk_3, jjj_3]);

                                            for (var iii = ii; iii < ii + L1_BLOCK_SIZE; iii += PART_MATRIX_DIM)
                                            {
                                                var iii_1 = iii + 1;
                                                var iii_2 = iii + 2;
                                                var iii_3 = iii + 3;

                                                var left4x4 = new Matrix4x4(left[iii, kkk], left[iii, kkk_1], left[iii, kkk_2], left[iii, kkk_3],
                                                                            left[iii_1, kkk], left[iii_1, kkk_1], left[iii_1, kkk_2], left[iii_1, kkk_3],
                                                                            left[iii_2, kkk], left[iii_2, kkk_1], left[iii_2, kkk_2], left[iii_2, kkk_3],
                                                                            left[iii_3, kkk], left[iii_3, kkk_1], left[iii_3, kkk_2], left[iii_3, kkk_3]);

                                                var product4x4 = left4x4 * right4x4;
                                                product[iii, jjj] += product4x4.M11; product[iii, jjj_1] += product4x4.M12; product[iii, jjj_2] += product4x4.M13; product[iii, jjj_3] += product4x4.M14;
                                                product[iii_1, jjj] += product4x4.M21; product[iii_1, jjj_1] += product4x4.M22; product[iii_1, jjj_2] += product4x4.M23; product[iii_1, jjj_3] += product4x4.M24;
                                                product[iii_2, jjj] += product4x4.M31; product[iii_2, jjj_1] += product4x4.M32; product[iii_2, jjj_2] += product4x4.M33; product[iii_2, jjj_3] += product4x4.M34;
                                                product[iii_3, jjj] += product4x4.M41; product[iii_3, jjj_1] += product4x4.M42; product[iii_3, jjj_2] += product4x4.M43; product[iii_3, jjj_3] += product4x4.M44;
                                            }
                                        }
                                    }

            // 端数処理
            for (var j = right.ColumnNum - j_rest; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = right.RowNum - k_rest; k < right.RowNum; k++)
                    for (var i = 0; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                    for (var i = left.RowNum - i_rest; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
        }

        public static void Mult_OpenBLAS(Matrix left, Matrix right, Matrix product)
        {
            cblas_sgemm(BLASOrder.ColMajor, BLASTranspose.NoTrans, BLASTranspose.NoTrans, left.RowNum, right.ColumnNum, right.RowNum,
                        1.0f, left.DATA, left.RowNum, right.DATA, right.RowNum, 0.0f, product.DATA, product.RowNum);
        }

        static float[] Transpose(float[] data, int rowNum, int colNum)
        {
            var ret = new float[data.Length];
            for (var i = 0; i < rowNum; i++)
                for (var j = 0; j < colNum; j++)
                    ret[j + i * colNum] = data[i + j * rowNum];
            return ret;
        }

        // OpenBLAS
        const string OPENBLAS_PATH = "OpenBLAS\\libopenblas.dll";

        public enum BLASOrder
        {
            RowMajor = 101,
            ColMajor = 102
        }

        public enum BLASTranspose
        {
            NoTrans = 111,
            Trans = 112,
            ConjTrans = 113,
            ConjNoTrans = 114
        }

        [DllImport(OPENBLAS_PATH)]
        static extern void cblas_saxpy(int n, float alpha, float[] x, int incX, float[] y, int incY);

        [DllImport(OPENBLAS_PATH)]
        static extern float cblas_sasum(int n, float[] x, int incX);

        [DllImport(OPENBLAS_PATH)]
        static extern float cblas_snrm2(int n, float[] x, int incX);

        [DllImport(OPENBLAS_PATH)]
        static extern float cblas_sscal(int n, float a, float[] x, int incX);

        [DllImport(OPENBLAS_PATH)]
        static extern void cblas_sgemm(
               BLASOrder order,
               BLASTranspose transa, BLASTranspose transb,
               int m, int n, int k,
               float alpha, float[] a, int ldA,
               float[] b, int ldB,
               float beta, float[] c, int ldC);
    }
}
