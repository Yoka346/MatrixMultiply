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
            PARALLEL_OPTIONS.MaxDegreeOfParallelism = Environment.ProcessorCount;
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

        public static void Mult_3(Matrix left, Matrix right, Matrix product)    // タイリング + 4段ループアンローリング
        {
            const int BLOCK_SIZE = 64;       
            const int UNROLL_STEP_NUM = 4;

            var j_rest = right.ColumnNum % BLOCK_SIZE;
            var k_rest = right.RowNum % BLOCK_SIZE;
            var i_rest = left.RowNum % BLOCK_SIZE;
            for (var jj = 0; jj < right.ColumnNum - j_rest; jj += BLOCK_SIZE)
                for (var kk = 0; kk < right.RowNum - k_rest; kk += BLOCK_SIZE)
                    for (var ii = 0; ii < left.RowNum - i_rest; ii += BLOCK_SIZE)
                    {
                        for (var dj = 0; dj < BLOCK_SIZE; dj += UNROLL_STEP_NUM)
                        {
                            var j = jj + dj;
                            var j_1 = j + 1;
                            var j_2 = j + 2;
                            var j_3 = j + 3;
                            for (var dk = 0; dk < BLOCK_SIZE; dk += UNROLL_STEP_NUM)
                            {
                                var k = kk + dk;
                                var k_1 = k + 1;
                                var k_2 = k + 2;
                                var k_3 = k + 3;
                                var right_00 = right[k, j]; var right_01 = right[k_1, j]; var right_02 = right[k_2, j]; var right_03 = right[k_3, j];
                                var right_10 = right[k, j_1]; var right_11 = right[k_1, j_1]; var right_12 = right[k_2, j_1]; var right_13 = right[k_3, j_1];
                                var right_20 = right[k, j_2]; var right_21 = right[k_1, j_2]; var right_22 = right[k_2, j_2]; var right_23 = right[k_3, j_2];
                                var right_30 = right[k, j_3]; var right_31 = right[k_1, j_3]; var right_32 = right[k_2, j_3]; var right_33 = right[k_3, j_3];
                                for (var di = 0; di < BLOCK_SIZE; di++)
                                {
                                    var i = ii + di;
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
            const int BLOCK_SIZE = 64;         
            const int UNROLL_STEP_NUM = 4;

            var jBlockNum = right.ColumnNum / BLOCK_SIZE;
            var j_rest = right.ColumnNum % BLOCK_SIZE;
            var k_rest = right.RowNum % BLOCK_SIZE;
            var i_rest = left.RowNum % BLOCK_SIZE;
            Parallel.For(0, jBlockNum, PARALLEL_OPTIONS, (blockId) =>
            {
                var jj = blockId * BLOCK_SIZE;
                for (var kk = 0; kk < right.RowNum - k_rest; kk += BLOCK_SIZE)
                    for (var ii = 0; ii < left.RowNum - i_rest; ii += BLOCK_SIZE)
                    {
                        for (var dj = 0; dj < BLOCK_SIZE; dj += UNROLL_STEP_NUM)
                        {
                            var j = jj + dj;
                            var j_1 = j + 1;
                            var j_2 = j + 2;
                            var j_3 = j + 3;
                            for (var dk = 0; dk < BLOCK_SIZE; dk += UNROLL_STEP_NUM)
                            {
                                var k = kk + dk;
                                var k_1 = k + 1;
                                var k_2 = k + 2;
                                var k_3 = k + 3;
                                var right_00 = right[k, j]; var right_01 = right[k_1, j]; var right_02 = right[k_2, j]; var right_03 = right[k_3, j];
                                var right_10 = right[k, j_1]; var right_11 = right[k_1, j_1]; var right_12 = right[k_2, j_1]; var right_13 = right[k_3, j_1];
                                var right_20 = right[k, j_2]; var right_21 = right[k_1, j_2]; var right_22 = right[k_2, j_2]; var right_23 = right[k_3, j_2];
                                var right_30 = right[k, j_3]; var right_31 = right[k_1, j_3]; var right_32 = right[k_2, j_3]; var right_33 = right[k_3, j_3];
                                for (var di = 0; di < BLOCK_SIZE; di++)
                                {
                                    var i = ii + di;
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

        public static unsafe void Mult_4(Matrix left, Matrix right, Matrix product)    // タイリング + AVX
        {
            const int BLOCK_SIZE = 64;
            const int AVX_FLOAT_VEC_DIM = 8;

            var j_rest = right.ColumnNum % BLOCK_SIZE;
            var k_rest = right.RowNum % BLOCK_SIZE;
            var i_rest = left.RowNum % BLOCK_SIZE;

            fixed (float* leftPtr = left.DATA)
            fixed (float* productPtr = product.DATA)
            {
                for (var jj = 0; jj < right.ColumnNum - j_rest; jj += BLOCK_SIZE)
                    for (var kk = 0; kk < right.RowNum - k_rest; kk += BLOCK_SIZE)
                        for (var ii = 0; ii < left.RowNum - i_rest; ii += BLOCK_SIZE)
                        {
                            for (var dj = 0; dj < BLOCK_SIZE; dj++)
                            {
                                var j = jj + dj;
                                for (var dk = 0; dk < BLOCK_SIZE; dk++)
                                {
                                    var k = kk + dk;
                                    var rightReg = Vector256.Create(right[k, j]);
                                    float* p0 = &leftPtr[ii + k * left.RowNum];
                                    float* p1 = &productPtr[ii + j * product.RowNum];

                                    var leftReg_0 = Avx.LoadVector256(&p0[0]);
                                    var productReg_0 = Avx.LoadVector256(&p1[0]); 
                                    var leftReg_1 = Avx.LoadVector256(&p0[AVX_FLOAT_VEC_DIM]);
                                    var productReg_1 = Avx.LoadVector256(&p1[AVX_FLOAT_VEC_DIM]); 
                                    var leftReg_2 = Avx.LoadVector256(&p0[2 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_2 = Avx.LoadVector256(&p1[2 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_3 = Avx.LoadVector256(&p0[3 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_3 = Avx.LoadVector256(&p1[3 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_4 = Avx.LoadVector256(&p0[4 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_4 = Avx.LoadVector256(&p1[4 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_5 = Avx.LoadVector256(&p0[5 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_5 = Avx.LoadVector256(&p1[5 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_6 = Avx.LoadVector256(&p0[6 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_6 = Avx.LoadVector256(&p1[6 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_7 = Avx.LoadVector256(&p0[7 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_7 = Avx.LoadVector256(&p1[7 * AVX_FLOAT_VEC_DIM]);

                                    productReg_0 = Fma.MultiplyAdd(leftReg_0, rightReg, productReg_0);
                                    productReg_1 = Fma.MultiplyAdd(leftReg_1, rightReg, productReg_1);
                                    productReg_2 = Fma.MultiplyAdd(leftReg_2, rightReg, productReg_2);
                                    productReg_3 = Fma.MultiplyAdd(leftReg_3, rightReg, productReg_3);
                                    productReg_4 = Fma.MultiplyAdd(leftReg_4, rightReg, productReg_4);
                                    productReg_5 = Fma.MultiplyAdd(leftReg_5, rightReg, productReg_5);
                                    productReg_6 = Fma.MultiplyAdd(leftReg_6, rightReg, productReg_6);
                                    productReg_7 = Fma.MultiplyAdd(leftReg_7, rightReg, productReg_7);

                                    Avx.Store(&p1[0], productReg_0);
                                    Avx.Store(&p1[AVX_FLOAT_VEC_DIM], productReg_1);
                                    Avx.Store(&p1[2 * AVX_FLOAT_VEC_DIM], productReg_2);
                                    Avx.Store(&p1[3 * AVX_FLOAT_VEC_DIM], productReg_3);
                                    Avx.Store(&p1[4 * AVX_FLOAT_VEC_DIM], productReg_4);
                                    Avx.Store(&p1[5 * AVX_FLOAT_VEC_DIM], productReg_5);
                                    Avx.Store(&p1[6 * AVX_FLOAT_VEC_DIM], productReg_6);
                                    Avx.Store(&p1[7 * AVX_FLOAT_VEC_DIM], productReg_7);
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

        public static unsafe void ParallelMult_4(Matrix left, Matrix right, Matrix product)
        {
            const int BLOCK_SIZE = 64;
            const int AVX_FLOAT_VEC_DIM = 8;

            var jBlockNum = right.ColumnNum / BLOCK_SIZE;
            var j_rest = right.ColumnNum % BLOCK_SIZE;
            var k_rest = right.RowNum % BLOCK_SIZE;
            var i_rest = left.RowNum % BLOCK_SIZE;

            Parallel.For(0, jBlockNum, PARALLEL_OPTIONS, (blockId) =>
            {
                fixed (float* leftPtr = left.DATA)
                fixed (float* productPtr = product.DATA)
                {
                    var jj = blockId * BLOCK_SIZE;
                    for (var kk = 0; kk < right.RowNum - k_rest; kk += BLOCK_SIZE)
                        for (var ii = 0; ii < left.RowNum - i_rest; ii += BLOCK_SIZE)
                            for (var dj = 0; dj < BLOCK_SIZE; dj++)
                            {
                                var j = jj + dj;
                                for (var dk = 0; dk < BLOCK_SIZE; dk++)
                                {
                                    var k = kk + dk;
                                    var rightReg = Vector256.Create(right[k, j]);
                                    float* p0 = &leftPtr[ii + k * left.RowNum];
                                    float* p1 = &productPtr[ii + j * product.RowNum];

                                    var leftReg_0 = Avx.LoadVector256(&p0[0]);
                                    var productReg_0 = Avx.LoadVector256(&p1[0]);
                                    var leftReg_1 = Avx.LoadVector256(&p0[AVX_FLOAT_VEC_DIM]);
                                    var productReg_1 = Avx.LoadVector256(&p1[AVX_FLOAT_VEC_DIM]);
                                    var leftReg_2 = Avx.LoadVector256(&p0[2 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_2 = Avx.LoadVector256(&p1[2 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_3 = Avx.LoadVector256(&p0[3 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_3 = Avx.LoadVector256(&p1[3 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_4 = Avx.LoadVector256(&p0[4 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_4 = Avx.LoadVector256(&p1[4 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_5 = Avx.LoadVector256(&p0[5 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_5 = Avx.LoadVector256(&p1[5 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_6 = Avx.LoadVector256(&p0[6 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_6 = Avx.LoadVector256(&p1[6 * AVX_FLOAT_VEC_DIM]);
                                    var leftReg_7 = Avx.LoadVector256(&p0[7 * AVX_FLOAT_VEC_DIM]);
                                    var productReg_7 = Avx.LoadVector256(&p1[7 * AVX_FLOAT_VEC_DIM]);

                                    productReg_0 = Fma.MultiplyAdd(leftReg_0, rightReg, productReg_0);
                                    productReg_1 = Fma.MultiplyAdd(leftReg_1, rightReg, productReg_1);
                                    productReg_2 = Fma.MultiplyAdd(leftReg_2, rightReg, productReg_2);
                                    productReg_3 = Fma.MultiplyAdd(leftReg_3, rightReg, productReg_3);
                                    productReg_4 = Fma.MultiplyAdd(leftReg_4, rightReg, productReg_4);
                                    productReg_5 = Fma.MultiplyAdd(leftReg_5, rightReg, productReg_5);
                                    productReg_6 = Fma.MultiplyAdd(leftReg_6, rightReg, productReg_6);
                                    productReg_7 = Fma.MultiplyAdd(leftReg_7, rightReg, productReg_7);

                                    Avx.Store(&p1[0], productReg_0);
                                    Avx.Store(&p1[AVX_FLOAT_VEC_DIM], productReg_1);
                                    Avx.Store(&p1[2 * AVX_FLOAT_VEC_DIM], productReg_2);
                                    Avx.Store(&p1[3 * AVX_FLOAT_VEC_DIM], productReg_3);
                                    Avx.Store(&p1[4 * AVX_FLOAT_VEC_DIM], productReg_4);
                                    Avx.Store(&p1[5 * AVX_FLOAT_VEC_DIM], productReg_5);
                                    Avx.Store(&p1[6 * AVX_FLOAT_VEC_DIM], productReg_6);
                                    Avx.Store(&p1[7 * AVX_FLOAT_VEC_DIM], productReg_7);
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

        public static void Mult_OpenBLAS(Matrix left, Matrix right, Matrix product)
        {
            cblas_sgemm(BLASOrder.ColMajor, 
                        BLASTranspose.NoTrans, BLASTranspose.NoTrans, 
                        left.RowNum, right.ColumnNum, right.RowNum,
                        1.0f, left.DATA, left.RowNum, 
                        right.DATA, right.RowNum, 
                        0.0f, product.DATA, product.RowNum);
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
