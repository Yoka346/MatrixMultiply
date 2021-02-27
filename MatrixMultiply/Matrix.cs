using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace MatrixMultiply
{
    public class Matrix     // 行列クラス. 列優先で配置.
    {
        const float EPSILON = 1.0e-3f;
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
            const int STEP_SIZE = 4;

            var j_rest = right.ColumnNum % STEP_SIZE;
            var k_rest = right.RowNum % STEP_SIZE;

            for (var j = 0; j < right.ColumnNum - j_rest; j += STEP_SIZE)
            {
                var j_1 = j + 1;
                var j_2 = j + 2;
                var j_3 = j + 3;
                for (var k = 0; k < right.RowNum - k_rest; k += STEP_SIZE)
                {
                    var k_1 = k + 1;
                    var k_2 = k + 2;
                    var k_3 = k + 3;
                    for (var i = 0; i < left.RowNum; i++)
                    {
                        product[i, j] += left[i, k] * right[k, j] + left[i, k_1] * right[k_1, j] + left[i, k_2] * right[k_2, j] + left[i, k_3] * right[k_3, j];
                        product[i, j_1] += left[i, k] * right[k, j_1] + left[i, k_1] * right[k_1, j_1] + left[i, k_2] * right[k_2, j_1] + left[i, k_3] * right[k_3, j_1];
                        product[i, j_2] += left[i, k] * right[k, j_2] + left[i, k_1] * right[k_1, j_2] + left[i, k_2] * right[k_2, j_2] + left[i, k_3] * right[k_3, j_2];
                        product[i, j_3] += left[i, k] * right[k, j_3] + left[i, k_1] * right[k_1, j_3] + left[i, k_2] * right[k_2, j_3] + left[i, k_3] * right[k_3, j_3];
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
                    for (var i = 0; i < left.RowNum; i++)
                    {
                        product[i, j] += left[i, k] * right[k, j] + left[i, k_1] * right[k_1, j] + left[i, k_2] * right[k_2, j] + left[i, k_3] * right[k_3, j];
                        product[i, j_1] += left[i, k] * right[k, j_1] + left[i, k_1] * right[k_1, j_1] + left[i, k_2] * right[k_2, j_1] + left[i, k_3] * right[k_3, j_1];
                        product[i, j_2] += left[i, k] * right[k, j_2] + left[i, k_1] * right[k_1, j_2] + left[i, k_2] * right[k_2, j_2] + left[i, k_3] * right[k_3, j_2];
                        product[i, j_3] += left[i, k] * right[k, j_3] + left[i, k_1] * right[k_1, j_3] + left[i, k_2] * right[k_2, j_3] + left[i, k_3] * right[k_3, j_3];
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

        public static unsafe void Mult_3(Matrix left, Matrix right, Matrix product)    //  AVX
        {
            const int AVX_FLOAT_VEC_LEN = 256 / (sizeof(float) * 8);

            var i_rest = left.RowNum % AVX_FLOAT_VEC_LEN;

            for (var j = 0; j < right.ColumnNum; j++)
                for (var k = 0; k < right.RowNum; k++)
                {
                    var rightReg = Vector256.Create(right[k, j]);
                    for (var i = 0; i < left.RowNum - i_rest; i += AVX_FLOAT_VEC_LEN)
                    {
                        Vector256<float> leftReg;
                        fixed (float* leftPtr = &left.DATA[i + k * left.RowNum])
                            leftReg = Avx.LoadVector256(leftPtr);

                        fixed (float* productPtr = &product.DATA[i + j * product.RowNum])
                        {
                            var productReg = Avx.LoadVector256(productPtr);
                            productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                            Avx.Store(productPtr, productReg);
                        }
                    }

                    for (var i = left.RowNum - i_rest; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
                }
        }

        public static unsafe void ParallelMult_3(Matrix left, Matrix right, Matrix product)
        {
            const int AVX_FLOAT_VEC_LEN = 256 / (sizeof(float) * 8);

            var i_rest = left.RowNum % AVX_FLOAT_VEC_LEN;

            Parallel.For(0, right.ColumnNum, PARALLEL_OPTIONS, (j) =>
            {
                for (var k = 0; k < right.RowNum; k++)
                {
                    var rightReg = Vector256.Create(right[k, j]);
                    for (var i = 0; i < left.RowNum - i_rest; i += AVX_FLOAT_VEC_LEN)
                    {
                        Vector256<float> leftReg;
                        fixed (float* leftPtr = &left.DATA[i + k * left.RowNum])
                            leftReg = Avx.LoadVector256(leftPtr);

                        fixed (float* productPtr = &product.DATA[i + j * product.RowNum])
                        {
                            var productReg = Avx.LoadVector256(productPtr);
                            productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                            Avx.Store(productPtr, productReg);
                        }
                    }

                    for (var i = left.RowNum - i_rest; i < left.RowNum; i++)
                        product[i, j] += left[i, k] * right[k, j];
                }
            });
        }

        public static void Mult_4(Matrix left, Matrix right, Matrix product)    // キャッシュブロッキング
        {
            const int L2_BLOCK_SIZE = 128;      // L2キャッシュに192KiBのデータを載せる.
            const int L1_BLOCK_SIZE = 64;       // L1キャッシュに48KiBのデータを載せる.

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
                                            for (var iii = ii; iii < ii + L1_BLOCK_SIZE; iii++)
                                                product[iii, jjj] += left[iii, kkk] * right[kkk, jjj];

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

        public static void ParallelMult_4(Matrix left, Matrix right, Matrix product)
        {
            const int L2_BLOCK_SIZE = 128;      // L2キャッシュに192KiBのデータを載せる.
            const int L1_BLOCK_SIZE = 64;       // L1キャッシュに48KiBのデータを載せる.

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
                                            for (var iii = ii; iii < ii + L1_BLOCK_SIZE; iii++)
                                                product[iii, jjj] += left[iii, kkk] * right[kkk, jjj];
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

        public static void Mult_5(Matrix left, Matrix right, Matrix product)    // キャッシュブロッキング + 4段ループアンローリング
        {
            const int L2_BLOCK_SIZE = 128;      // L2キャッシュに192KiBのデータを載せる.
            const int L1_BLOCK_SIZE = 64;       // L1キャッシュに48KiBのデータを載せる.
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
                                            for (var iii = ii; iii < ii + L1_BLOCK_SIZE; iii++)
                                            {
                                                product[iii, jjj] += left[iii, kkk] * right[kkk, jjj] + left[iii, kkk_1] * right[kkk_1, jjj] + left[iii, kkk_2] * right[kkk_2, jjj] + left[iii, kkk_3] * right[kkk_3, jjj];
                                                product[iii, jjj_1] += left[iii, kkk] * right[kkk, jjj_1] + left[iii, kkk_1] * right[kkk_1, jjj_1] + left[iii, kkk_2] * right[kkk_2, jjj_1] + left[iii, kkk_3] * right[kkk_3, jjj_1];
                                                product[iii, jjj_2] += left[iii, kkk] * right[kkk, jjj_2] + left[iii, kkk_1] * right[kkk_1, jjj_2] + left[iii, kkk_2] * right[kkk_2, jjj_2] + left[iii, kkk_3] * right[kkk_3, jjj_2];
                                                product[iii, jjj_3] += left[iii, kkk] * right[kkk, jjj_3] + left[iii, kkk_1] * right[kkk_1, jjj_3] + left[iii, kkk_2] * right[kkk_2, jjj_3] + left[iii, kkk_3] * right[kkk_3, jjj_3];
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

        public static void ParallelMult_5(Matrix left, Matrix right, Matrix product)
        {
            const int L2_BLOCK_SIZE = 128;      // L2キャッシュに192KiBのデータを載せる.
            const int L1_BLOCK_SIZE = 64;       // L1キャッシュに48KiBのデータを載せる.
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
                                                product[iii, jjj] += left[iii, kkk] * right[kkk, jjj] + left[iii, kkk_1] * right[kkk_1, jjj] + left[iii, kkk_2] * right[kkk_2, jjj] + left[iii, kkk_3] * right[kkk_3, jjj];
                                                product[iii, jjj_1] += left[iii, kkk] * right[kkk, jjj_1] + left[iii, kkk_1] * right[kkk_1, jjj_1] + left[iii, kkk_2] * right[kkk_2, jjj_1] + left[iii, kkk_3] * right[kkk_3, jjj_1];
                                                product[iii, jjj_2] += left[iii, kkk] * right[kkk, jjj_2] + left[iii, kkk_1] * right[kkk_1, jjj_2] + left[iii, kkk_2] * right[kkk_2, jjj_2] + left[iii, kkk_3] * right[kkk_3, jjj_2];
                                                product[iii, jjj_3] += left[iii, kkk] * right[kkk, jjj_3] + left[iii, kkk_1] * right[kkk_1, jjj_3] + left[iii, kkk_2] * right[kkk_2, jjj_3] + left[iii, kkk_3] * right[kkk_3, jjj_3];
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

        public static unsafe void Mult_6(Matrix left, Matrix right, Matrix product)    // キャッシュブロッキング + AVX
        {
            const int BLOCK_2 = 128;      
            const int BLOCK_1 = 64;     
            const int AVX_FLOAT_VEC_DIM = 256 / (sizeof(float) * 8);

            var jBlockNum = right.ColumnNum / BLOCK_2;
            var j_rest = right.ColumnNum % BLOCK_2;
            var k_rest = right.RowNum % BLOCK_2;
            var i_rest = left.RowNum % BLOCK_2;
            for (var j = 0; j < right.ColumnNum - j_rest; j += BLOCK_2)
                for (var k = 0; k < right.RowNum - k_rest; k += BLOCK_2)
                    for (var i = 0; i < left.RowNum - i_rest; i += BLOCK_2)
                        for (var jj = j; jj < j + BLOCK_2; jj += BLOCK_1)
                            for (var kk = k; kk < k + BLOCK_2; kk += BLOCK_1)
                                for (var ii = i; ii < i + BLOCK_2; ii += BLOCK_1)
                                    for (var jjj = jj; jjj < jj + BLOCK_1; jjj++)
                                        for (var kkk = kk; kkk < kk + BLOCK_1; kkk++)
                                        {
                                            fixed (float* leftPtr = &left.DATA[ii + kkk * left.RowNum])
                                            fixed (float* productPtr = &product.DATA[ii + jjj * product.RowNum])
                                            {
                                                var leftReg = Avx.LoadVector256(leftPtr);
                                                var rightReg = Vector256.Create(right[kkk, jjj]);
                                                var productReg = Avx.LoadVector256(productPtr);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 2 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 2 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 2 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 3 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 3 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 3 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 4 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 4 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 4 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 5 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 5 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 5 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 6 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 6 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 6 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 7 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 7 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 7 * AVX_FLOAT_VEC_DIM, productReg);
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

        public static unsafe void ParallelMult_6(Matrix left, Matrix right, Matrix product)    
        {
            const int L2_BLOCK_SIZE = 128;      // L2キャッシュに192KiBのデータを載せる.
            const int L1_BLOCK_SIZE = 64;       // L1キャッシュに48KiBのデータを載せる.
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
                                                var leftReg = Avx.LoadVector256(leftPtr);
                                                var rightReg = Vector256.Create(right[kkk, jjj]);
                                                var productReg = Avx.LoadVector256(productPtr);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 2 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 2 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 2 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 3 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 3 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 3 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 4 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 4 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 4 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 5 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 5 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 5 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 6 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 6 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 6 * AVX_FLOAT_VEC_DIM, productReg);

                                                leftReg = Avx.LoadVector256(leftPtr + 7 * AVX_FLOAT_VEC_DIM);
                                                productReg = Avx.LoadVector256(productPtr + 7 * AVX_FLOAT_VEC_DIM);
                                                productReg = Fma.MultiplyAdd(leftReg, rightReg, productReg);
                                                Avx.Store(productPtr + 7 * AVX_FLOAT_VEC_DIM, productReg);
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
