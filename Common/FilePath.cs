using System;
using System.IO;

namespace Common
{
    public class FilePath
    {
        public static string GetAbsolutePath(Type programType, string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(programType.Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}