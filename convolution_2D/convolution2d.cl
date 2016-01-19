__kernel void convolution_2d_gpu_global(__global float * matrix, __global float * mask, int matrix_size, int mask_size,
                          __global float * result)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   if (x >= matrix_size || y >= matrix_size)
       return;

   float tmp = 0;

   for (int i = 0; i < mask_size; ++i)
   {
       for (int j = 0; j < mask_size; ++j)
	   {
	        int cx = x + i - mask_size / 2;
            int cy = y + j - mask_size / 2;
            if (cx >= 0 && cx < matrix_size && cy >= 0 && cy < matrix_size)
                tmp += matrix[matrix_size * cx + cy] * mask[mask_size * i + j];
	   }
   }

   result[matrix_size * x + y] = tmp;
}

