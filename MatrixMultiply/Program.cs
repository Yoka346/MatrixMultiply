using System;
using System.Diagnostics;

namespace MatrixMultiply
{
    static class Program
    {
        static readonly Random RAND = new Random();
        const int SAMPLE_NUM = 100;

        static void Main(string[] args)
        {
#if RELEASE
            const int ROW_NUM = 1024;
            const int COLUMN_NUM = 1024;
#elif DEBUG
            const int ROW_NUM = 256;
            const int COLUMN_NUM = 256;
#endif

            var left = new Matrix(ROW_NUM, COLUMN_NUM);
            var right = new Matrix(COLUMN_NUM, ROW_NUM);
            var productTest = new Matrix(ROW_NUM, ROW_NUM);
            left.InitByRandNum();
            right.InitByRandNum();
            Matrix.ParallelMult_1(left, right, productTest);

            DoTest("Simple", Matrix.Mult_0, left, right, productTest);
            DoTest("Simple parallel", Matrix.ParallelMult_0, left, right, productTest);
            DoTest("Swap loop ordering", Matrix.Mult_1, left, right, productTest);
            DoTest("Swap loop ordering parallel", Matrix.ParallelMult_1, left, right, productTest);
            DoTest("4 step loop unrolling", Matrix.Mult_2, left, right, productTest);
            DoTest("4 step loop unrolling parallel", Matrix.ParallelMult_2, left, right, productTest);
            DoTest("Cache blocking + 4step loop unrolling", Matrix.Mult_3, left, right, productTest);
            DoTest("Cache blocking + 4step loop unrolling parallel", Matrix.ParallelMult_3, left, right, productTest);
            DoTest("Cache blocking + AVX", Matrix.Mult_4, left, right, productTest);
            DoTest("Cache blocking + AVX parallel", Matrix.ParallelMult_4, left, right, productTest);
            DoTest("Cache blocking + Matrix4x4", Matrix.Mult_5, left, right, productTest);
            DoTest("OpenBLAS", Matrix.Mult_OpenBLAS, left, right, productTest);
        }

        static void DoTest(string testLabel, Action<Matrix, Matrix, Matrix> func, Matrix left, Matrix right, Matrix productTest)
        {
            Console.WriteLine($"Begin Test: {testLabel}");

            var product = new Matrix(left.RowNum, right.ColumnNum);
            func(left, right, product);
            if (!product.Equals(productTest)) 
            {
                Console.WriteLine("Failed: Invalid product.");
                Console.WriteLine($"End Test: {testLabel}");
                return;
            }

            var sw = new Stopwatch();
            sw.Start();
            for (var i = 0; i < SAMPLE_NUM; i++)
                func(left, right, product);
            sw.Stop();

            var meanTime = (float)sw.ElapsedMilliseconds / SAMPLE_NUM;
            Console.WriteLine($"Success: Mean of ellapsed {meanTime} ms\n\t{(2.0f * right.ColumnNum * right.RowNum * left.RowNum) / (meanTime * 1.0e+6f)} GFLOPS");
            Console.WriteLine($"End Test: {testLabel}\n");
        }

        static void InitByRandNum(this Matrix matrix)
        {
            for (var j = 0; j < matrix.ColumnNum; j++)
                for (var i = 0; i < matrix.RowNum; i++)
                    matrix[i, j] = (float)RAND.NextDouble();
        }
    }
}
