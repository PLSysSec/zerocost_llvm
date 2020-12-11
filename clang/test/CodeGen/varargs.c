// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -target-feature +separate-stack-seg -emit-llvm -o - %s | FileCheck -check-prefix=SEPARATE-SS %s

// PR6433 - Don't crash on va_arg(typedef).
typedef double gdouble;
void focus_changed_cb () {
    __builtin_va_list pa;
    double mfloat;
    mfloat = __builtin_va_arg((pa), gdouble);
}

void vararg(int, ...);
void function_as_vararg() {
  // CHECK: define {{.*}}function_as_vararg
  // CHECK-NOT: llvm.trap
  vararg(0, focus_changed_cb);
}

void vla(int n, ...)
{
  __builtin_va_list ap;
  void *p;
  p = __builtin_va_arg(ap, typeof (int (*)[++n])); // CHECK: add nsw i32 {{.*}}, 1
  // SEPARATE-SS: load i32*, i32* addrspace(258)* {{.*}}
}
