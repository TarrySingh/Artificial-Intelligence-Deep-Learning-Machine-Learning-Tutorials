/*
[The "BSD licence"]
Copyright (c) 2005-2007 Kunle Odutola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


namespace Antlr.Runtime.Misc
{
	using System;
	using FileInfo = System.IO.FileInfo;
	using DirectoryInfo = System.IO.DirectoryInfo;
	using StreamWriter = System.IO.StreamWriter;
	using Path = System.IO.Path;
	using IOException = System.IO.IOException;

	/// <summary>Stats routines needed by profiler etc...</summary>
	/// <remarks>
	/// Note that these routines return 0.0 if no values exist in X[]
	/// which is not "correct" but, it is useful so I don't generate NaN
	/// in my output
	/// </remarks>
	public class Stats
	{
		/// <summary>Compute the sample (unbiased estimator) standard deviation</summary>
		/// <remarks>
		/// The computation follows:
		///    Computing Deviations: Standard Accuracy
		///    Tony F. Chan and John Gregg Lewis
		///    Stanford University
		///    Communications of ACM September 1979 of Volume 22 the ACM Number 9
		///   
		/// The "two-pass" method from the paper; supposed to have better
		/// numerical properties than the textbook summation/sqrt.  To me
		/// this looks like the textbook method, but I ain't no numerical
		/// methods guy.
		/// </remarks>
		public static double Stddev(int[] X)
		{
			int m = X.Length;
			if (m <= 1)
			{
				return 0;
			}
			double xbar = Avg(X);
			double s2 = 0.0;
			for (int i = 0; i < m; i++)
			{
				s2 += (X[i] - xbar) * (X[i] - xbar);
			}
			s2 = s2 / (m - 1);
			return Math.Sqrt(s2);
		}

		/// <summary>Compute the sample mean</summary>
		public static double Avg(int[] X)
		{
			double xbar = 0.0;
			int m = X.Length;
			if (m == 0)
			{
				return 0;
			}
			for (int i = 0; i < m; i++)
			{
				xbar += X[i];
			}
			if (xbar >= 0.0)
			{
				return xbar / m;
			}
			return 0.0;
		}

		public static int Min(int[] X)
		{
			int min = Int32.MaxValue;
			int m = X.Length;
			if (m == 0)
			{
				return 0;
			}
			for (int i = 0; i < m; i++)
			{
				if (X[i] < min)
				{
					min = X[i];
				}
			}
			return min;
		}

		public static int Max(int[] X)
		{
			int max = Int32.MinValue;
			int m = X.Length;
			if (m == 0)
			{
				return 0;
			}
			for (int i = 0; i < m; i++)
			{
				if (X[i] > max)
				{
					max = X[i];
				}
			}
			return max;
		}

		public static int Sum(int[] X)
		{
			int s = 0;
			int m = X.Length;
			if (m == 0)
			{
				return 0;
			}
			for (int i = 0; i < m; i++)
			{
				s += X[i];
			}
			return s;
		}

		public static void WriteReport(string filename, string data)
		{
			string absoluteFilename = GetAbsoluteFileName(filename);
			FileInfo f = new FileInfo(absoluteFilename);
			f.Directory.Create(); // ensure parent dir exists
			// write file
			try
			{
				StreamWriter w = new StreamWriter(f.FullName, true); // append
				w.WriteLine(data);
				w.Close();
			}
			catch (IOException ioe)
			{
				ErrorManager.InternalError("can't write stats to " + absoluteFilename,
										   ioe);
			}
		}

		public static string GetAbsoluteFileName(string filename)
		{
			return Path.Combine(
				Path.Combine(Environment.CurrentDirectory, Constants.ANTLRWORKS_DIR),
				filename);
		}
	}
}