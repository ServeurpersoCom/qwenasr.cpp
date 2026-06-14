#pragma once
// utf8.h: portable UTF-8 boundary for Windows. Inside the project everything
// is UTF-8; this header bridges to the Windows-native UTF-16 APIs at the
// three places where the OS forces a recode: argv (CRT decodes from CP_ACP),
// fopen (CRT does too), and any direct Win32 *A call (CreateFileA etc.).
// POSIX is UTF-8 by convention and every helper degrades to a passthrough.

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
// clang-format off
// windows.h must precede shellapi.h: shellapi.h pulls in types (HDROP,
// DECLSPEC_IMPORT, ...) declared by windows.h. SortIncludes would alphabetise
// the pair and break the build, so the off/on pragmas pin the order.
#    include <windows.h>
#    include <shellapi.h>
// clang-format on
#include <string>
#endif

#if defined(_WIN32)
// Decode a UTF-8 C string to a wide string (UTF-16). Empty on null or empty
// input. Used by every Win32 path-taking call site (CreateFileW, MoveFileW
// and friends) wherever the project still needs the raw wide form.
static std::wstring utf8_to_wide(const char *s) {
  if (!s || !*s) {
    return std::wstring();
  }
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
  if (n <= 1) {
    return std::wstring();
  }
  std::wstring w((size_t)(n - 1), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
  return w;
}
#endif

// First line of main(). Rebuilds argv as UTF-8 from the wide command line
// and switches console I/O to CP_UTF8 so fprintf with non-ASCII bytes
// renders correctly without an explicit chcp 65001. POSIX no-op.
//
// The replacement argv is allocated once and leaked for the process
// lifetime: the OS reclaims it on exit, and the top of main is the only
// site that would ever free it.
static void utf8_init(int *argc, char ***argv) {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  int wargc = 0;
  LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
  if (!wargv) {
    return;
  }

  char **new_argv = (char **)malloc(sizeof(char *) * (size_t)(wargc + 1));
  for (int i = 0; i < wargc; i++) {
    int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
    new_argv[i] = (char *)malloc((size_t)n);
    WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, new_argv[i], n, NULL, NULL);
  }
  new_argv[wargc] = NULL;
  LocalFree(wargv);

  *argc = wargc;
  *argv = new_argv;
#else
  (void)argc;
  (void)argv;
#endif
}

// UTF-8 aware fopen. POSIX passthrough. Windows decodes the UTF-8 path and
// mode to UTF-16 and calls _wfopen, the only fopen variant on MSVC that
// bypasses CP_ACP.
static FILE *utf8_fopen(const char *path, const char *mode) {
#if defined(_WIN32)
  std::wstring wpath = utf8_to_wide(path);
  std::wstring wmode = utf8_to_wide(mode);
  return _wfopen(wpath.c_str(), wmode.c_str());
#else
  return fopen(path, mode);
#endif
}
