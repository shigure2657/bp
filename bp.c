#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define NUM_LEARN 10000   // 学習の繰り返し回数をここで指定する。
#define NUM_SAMPLE 6	// 訓練データのサンプル数。
#define NUM_INPUT 3		// 入力ノード数。
#define NUM_HIDDEN 5	// 中間層（隠れ層）の素子数。
#define NUM_OUTPUT 1	// 出力素子数。
#define EPSILON 0.05	 // 学習時の重み修正の程度を決める。
#define THRESHOLD_ERROR 0.01 // 学習誤差がこの値以下になるとプログラムは停止する。

int tx[NUM_SAMPLE][NUM_INPUT], ty[NUM_SAMPLE][NUM_OUTPUT];	// 訓練データを格納する配列。
float x[NUM_INPUT+1], h[NUM_HIDDEN+1], y[NUM_OUTPUT];			// 閾値表現用に１つ余分に確保。
float w1[NUM_INPUT+1][NUM_HIDDEN], w2[NUM_HIDDEN+1][NUM_OUTPUT]; 	// 閾値表現用に１つ余分に確保。
float h_back[NUM_HIDDEN+1], y_back[NUM_OUTPUT];	 // 隠れ素子、出力素子における逆伝搬量。
float sigmoid_table[100]; // 計算高速化のためのシグモイド関数値のテーブル。

main()
{
    int ilearn, isample, i, j;
    float net_input, error, max_error, epsilon, seed;
    int inet_input;
    FILE *stream;

    epsilon = (float)EPSILON;

// 計算高速化のためシグモイド関数のテーブルの作成。ゲインは1に固定。
   for ( i = 0; i < 100; i++ ) 
     {
      sigmoid_table[i] = (float)(1.0 / (1.0 + exp( (double)(i-50) * -0.2 ) ) );
     }

// 訓練データのファイルからの読み込み。
   stream = fopen( "training.dat", "r" );
   if( stream == NULL )
     {
      printf( "ファイルtraining.datをオープンできません。\n" );
      exit(0);
     }
   else
     {
      /* ファイルからデータを読みこむ */
      for ( isample = 0; isample < NUM_SAMPLE; isample++ )
        {
         for ( i = 0; i < NUM_INPUT; i++ ) 
	   {
            fscanf( stream, "%d", &tx[isample][i] );
	   }
         for ( i = 0; i < NUM_OUTPUT; i++ )
           {
            fscanf( stream, "%d", &ty[isample][i] );
	   }
	 }

      /* 読み込んだデータの表示 */
      for ( isample = 0; isample < NUM_SAMPLE; isample++ )
	 {
          printf( "訓練データ NO. %d :   ", isample+1 );
          printf( "入力： " );
          for ( i = 0; i < NUM_INPUT; i++ )
	    {
             printf( " %d ", tx[isample][i] );
            }
          printf( "出力： " );
          for ( i = 0; i < NUM_OUTPUT; i++ )
	    {
             printf( "%d", ty[isample][i] );
	    }
          printf( "\n" );
         }

      fclose( stream );
   }
// 重み初期値の設定。初期値は全て 0 にしないこと。
    seed = (float) 1;
    for ( i = 0; i < NUM_INPUT+1; i++ )
       for ( j = 0; j < NUM_HIDDEN; j++ )
         {
          seed = seed * (float)-1;
          w1[i][j] = seed;
         }
    for ( i = 0; i < NUM_HIDDEN+1; i++ )
       for ( j = 0; j < NUM_OUTPUT; j++ )
	  {
	   seed = seed * (float)-1;
	   w2[i][j] = seed;
	  }

// 学習の繰り返しループ。
    for ( ilearn = 0; ilearn < NUM_LEARN; ilearn++ )
      {
// 訓練データに関するループ。
       max_error = 0;
       for ( isample = 0; isample < NUM_SAMPLE; isample++ )
	 {
          // 順方向の動作。
          // 訓練データに従って、ネットワークへの入力を設定する。
          for ( i = 0; i < NUM_INPUT; i++ )
	    {
             x[i] = tx[isample][i];
	    }
	  // 閾値用に x[NUM_INPUT] = 1.0 とする。
	  x[NUM_INPUT] = (float)1.0;

	  // 隠れ素子値の計算。
          for ( j = 0; j < NUM_HIDDEN; j++ )
	    {
	     net_input = 0;
             for ( i = 0; i < NUM_INPUT+1; i++ )
	       {
                net_input = net_input + w1[i][j] * x[i];
	       }
	   // テーブルに格納したシグモイド関数値の利用。
	   inet_input = (int)(net_input * 5) + 50;
	   if ( inet_input > 99 ) inet_input = 99;
	   else if ( inet_input < 0 ) inet_input = 0;
	   h[j] = sigmoid_table[inet_input];
	  }
	  h[NUM_HIDDEN] = (float)1.0;

	  // 出力素子値の計算。
          for ( j = 0; j < NUM_OUTPUT; j++ )
            {
             net_input = 0;
             for ( i = 0; i < NUM_HIDDEN+1; i++ )
	       {
                net_input = net_input + w2[i][j] * h[i];
	       }
	   // テーブルに格納したシグモイド関数値の利用。
	   inet_input = (int)(net_input * 5) + 50;
	   if ( inet_input > 99 ) inet_input = 99;
	   else if ( inet_input < 0 ) inet_input = 0;
	   y[j] = sigmoid_table[inet_input];
	  }

          // 誤差の評価。
	  error = 0;
          for ( j = 0; j < NUM_OUTPUT; j++ )
	   {
	    error = error + (ty[isample][j] - y[j]) * (ty[isample][j] - y[j]);
	   }
	  error = error / (float)NUM_OUTPUT;
	  if ( error > max_error ) max_error = error;
	  printf( "学習回数 = %d, 訓練データNO. = %d, 誤差 = %f \n", ilearn, isample+1, error ); 

          // 逆方向の動作。
	  // 出力層素子の逆伝搬時の動作。
          for ( j = 0; j < NUM_OUTPUT; j++ )
	    {
	     y_back[j] = (y[j] - ty[isample][j]) * ((float)1.0 - y[j]) * y[j];
            }
	  // 隠れ層素子の逆伝搬時の動作。
          for ( i = 0; i < NUM_HIDDEN; i++ )
	   {
	    net_input = 0;
            for ( j = 0; j < NUM_OUTPUT; j++ )
	      {
               net_input = net_input + w2[i][j] * y_back[j];
	      }
	    h_back[i] = net_input * ((float)1.0 - h[i]) * h[i];
	   }
	  // 重みの修正。
          for ( i = 0; i < NUM_INPUT+1; i++ )
             for ( j = 0; j < NUM_HIDDEN; j++ )
	        w1[i][j] = w1[i][j] - epsilon * x[i] * h_back[j];
	  for ( i = 0; i < NUM_HIDDEN+1; i++ )
             for ( j = 0; j < NUM_OUTPUT; j++ )
	        w2[i][j] = w2[i][j] - epsilon * h[i] * y_back[j];

	 } // 訓練データに関するループ。
       if ( max_error < THRESHOLD_ERROR) break;
      } // 学習の繰り返しループ。
}
