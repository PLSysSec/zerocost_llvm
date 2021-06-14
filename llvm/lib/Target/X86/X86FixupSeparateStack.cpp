//===----- X86FixupSeparateStack.cpp - Fixup stack memory accesses --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that emits segment override prefixes to help support
// separate stack and data segments for X86-32.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "X86MachineFunctionInfo.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define FIXUPSEPSTK_DESC "X86 Separate Stack Fixup"
#define FIXUPSEPSTK_NAME "x86-fixup-separate-stack"

static cl::list<std::string> FlatMemFuncs(
    "sep-stk-seg-flat-mem-func",
    cl::Hidden,
    cl::desc("Specify a function that operates with a flat memory model, even "
             "though the target machine uses a separate stack segment.  For "
             "example, this may be applied to runtime initialization functions "
             "that run before segment restrictions are activated."));

namespace {
enum class RegValueType {
  StackPointer,
  NotStackPointer,
  /// The register may or may not contain a stack pointer, depending on how
  /// previous instructions executed.  For example, the register may have been
  /// defined by a conditional move instruction with one stack pointer operand
  /// and one operand with a different type.
  Ambiguous
};

/// Records information about registers that directly or indirectly are used in
/// memory operands within a basic block.  It is used to distinguish registers
/// that point to stack memory from those that do not or that are ambiguous.
class AddrRegs {
private:
  const X86RegisterInfo *TRI;

public:
  DenseMap<unsigned, RegValueType> RegTypes;

  AddrRegs(const X86RegisterInfo *TRI_) : TRI(TRI_) {}

  void setRegType(unsigned Reg, RegValueType type);

  RegValueType getRegType(unsigned Reg) const;
};

/// Requirements for how registers that are directly or indirectly used in
/// memory operands are configured.
///
/// Each basic block is associated with one AddrRegReqs object.  That object
/// may also contain information about the successors of the basic block with
/// which it is associated.  When a successor uses a register, the requirements
/// for that register get chained back through predecessors' AddrRegReqs
/// if the register value is derived from register values in predecessors.
class AddrRegReqs {
private:
  const X86RegisterInfo *TRI;

  /// The collection of AddrRegReqs objects that is associated with a function
  /// is structured as a directed graph that may contain cycles.  This set of
  /// Predecessors points to the AddrRegReqs objects corresponding to all of the
  /// basic blocks that immediately precede the one associated with this object
  /// and that have already been visited or are currently being visited by this
  /// pass.
  SmallPtrSet<AddrRegReqs *, 16> Predecessors;
  /// The original status of the registers at the time this object branched from
  /// its first predecessor, according to the traversal order of this pass.
  const AddrRegs OrigRegs;

  /// If a maps to B, then the original value of every register b (according to
  /// the definition of OrigRegs) in B influenced the current value of a.  See
  /// X86FixupSeparateStack::mayComputePtr for details.
  ///
  /// At the beginning of each basic block, the value of each physical register
  /// is implicitly derived from the original value of that register.  It is as
  /// though Derivatives contains (a, {a}) for each physical register a.
  /// Inserting an explicit entry (a, X) in Derivatives for some (possibly
  /// empty) set X cancels the implicit derivation for a.
  ///
  /// Note that the presence of an entry (a, B) in Derivatives does NOT
  /// automatically indicate that a points to stack data.  It only indicates
  /// that if all registers b in B pointed to stack data when this basic block
  /// was entered, then a points to stack data at the current instruction.
  ///
  /// The entry (ESP, {ESP}) is always implicitly (and never explicitly) present
  /// in Derivatives.
  ///
  /// Some possible derivations may be omitted from Derivatives.  See
  /// X86FixupSeparateStack::mayComputePtr for details.
  DenseMap<unsigned, DenseSet<unsigned> > Derivatives;
  /// Root registers that are used directly or indirectly in memory operands in
  /// this basic block and/or in its successors.
  DenseSet<unsigned> UsedInMemOp;

  /// \brief Lookup any root registers from which the current value of To may
  /// be derived.
  ///
  /// \returns The set of registers Roots such that an implicit or
  /// explicit entry (To, Roots) is currently present in Derivatives.
  DenseSet<unsigned> lookupRoots(unsigned To) const;

public:
  /// \brief Construct an object with no initial predecessor.  One or more
  /// predecessors may still be added later with addPredecessor.
  AddrRegReqs(const X86RegisterInfo *TRI_) : TRI(TRI_), OrigRegs(TRI_) {}
  /// \brief Construct an object with one initial predecessor.  Initialize
  /// OrigRegs to represent the register configuration at the end of
  /// Predecessor, since this is only invoked after Predecessor has been
  /// processed.
  AddrRegReqs(AddrRegReqs *Predecessor);

  /// \brief Extract the register configuration represented by the current state
  /// of this object.  This is based on the contents of OrigRegs and
  /// Derivatives.
  AddrRegs extractRegs();

  /// \brief Register a new predecessor for this object.  This involves the
  /// following steps:
  /// 1. Insert Predecessor into Predecessors so that any time useInMemOp is
  ///    invoked on this object, it may also invoke Predecessor->useInMemOp as
  ///    necessary.
  /// 2. Invoke Predecessor->useInMemOp for each register in UsedInMemOp, since
  ///    the basic block corresponding to this object uses those registers even
  ///    though the instructions that are in the predecessor basic block may not
  ///    have used those registers.
  void addPredecessor(AddrRegReqs *Predecessor);

  /// \brief Record that To is derived from From in Derivatives.
  void derive(unsigned From, unsigned To);

  /// \brief Record that To is redefined by some other type of operation besides
  /// those used to derive pointers.
  void markNotDerived(unsigned To);

  /// \brief Record that Reg is used as a base address register in a memory
  /// operand and check that the requirement for this to be a stack or data
  /// pointer is met.
  ///
  /// This routine is invoked recursively for each predecessor of this object,
  /// and there may even be cycles.  However, the recursion will eventually
  /// terminate.  The routine is only invoked recursively if an entry is added
  /// to this->UsedInMemOp.  The only entries that can be added are those
  /// computed from the contents of Derivatives and OrigRegs.  The resultant set
  /// is a subset of the set of physical registers, which is finite.  Entries
  /// are never removed from UsedInMemOp by this routine.  Thus, this routine
  /// will eventually run out of entries to be added to UsedInMemOp.
  ///
  /// \param Reg Register that is used as a base address register in a memory
  ///            operand.
  ///
  /// \return Type of value in Reg at current instruction.
  RegValueType useInMemOp(unsigned Reg);

  /// \brief Return type of value in Reg at the current instruction.
  RegValueType getRegType(unsigned Reg) const;
};

/// Requirements for how register spills of pointers to the stack segment are
/// arranged in the frame.
class StackPtrSpillReqs {
private:
  std::set<StackPtrSpillReqs *> Predecessors;

  /// Frame slots that have been used in the associated basic block and/or its
  /// successors to fill registers with stack pointers or other data.
  IntervalMap<int64_t, bool> Demand;

  /// Frame slots that have stack pointers or other data spilled into them by
  /// instructions in the basic block associated with this object.  The contents
  /// of this map change as the basic block is traversed to reflect the supply
  /// at the current instruction in the basic block.
  IntervalMap<int64_t, bool> Supply;

  static IntervalMap<int64_t, bool>::Allocator IntervalMapAlloc;

  /// True if this object is associated with the function entrypoint.
  bool Top;

  enum class DemandResponse { Unsatisfied, SatisfiedSP, SatisfiedNonSP };

  /// \brief Check whether a spill slot currently contains a stack pointer or
  /// other data and register this demand.
  DemandResponse demand(int64_t Start, uint64_t Size,
                        SmallPtrSet<StackPtrSpillReqs *, 16>& Demanders);

public:
  /// \brief Construct an object with no initial predecessor.  One or more
  /// predecessors may still be added later with addPredecessor.
  StackPtrSpillReqs() :
    Demand(IntervalMapAlloc), Supply(IntervalMapAlloc), Top(true) {}
  /// \brief Construct an object with one initial predecessor.
  StackPtrSpillReqs(StackPtrSpillReqs *Predecessor);

  /// \brief Register a new predecessor for this object.  This involves the
  /// following steps:
  /// 1. Insert Predecessor into Predecessors so that any time demand is
  ///    invoked on this object, it may also invoke Predecessor->demand as
  ///    necessary.
  /// 2. Invoke Predecessor->demand for each frame slot demanded by this set of
  ///    requirements.
  void addPredecessor(StackPtrSpillReqs *Predecessor);

  /// \brief Record that either a stack pointer or other data is spilled into
  /// the specified slot by the current instruction being processed in the basic
  /// block associated with this object.
  void supply(bool SP, int64_t Start, uint64_t Size);

  /// \brief Check whether the specified slot most recently had stack pointer
  /// data spilled into it.
  bool isSPSpill(int64_t Start, uint64_t Size);
};

/// This pass performs a depth-first traversal of each function, visiting each
/// basic block once.
///
/// To understand how this pass determines whether its transformation is
/// correct, it may be helpful to think of each basic block as an electronic
/// circuit with 12V and 5V inputs.  Those inputs take the form of register
/// values fed into basic blocks.  Think of pointers to stack data as 12V
/// inputs and other register values as 5V inputs.
///
/// Continuing this analogy, the problem that this pass seeks to solve is to
/// determine whether certain critical circuit elements (analogous to memory
/// operands) are each always wired to 12V or 5V.  This pass considers each
/// instruction to be analogous to a circuit element.  Of course, each basic
/// block may be invoked after multiple predecessors.  Think of this like a
/// circuit that can have multiple copies of a particular circuit block
/// comprising circuit elements such as resistors and capacitors, and all
/// replicas of a particular element must use the same voltage inputs.
///
/// It is necessary to analyze the control flow between basic blocks to make
/// sure that for each base address register used in each memory operand, the
/// register consistently refers to either stack data or non-stack data for
/// every control flow that contains the corresponding instruction.  For
/// efficiency, this pass only traverses each basic block once.  The
/// connectivity check is performed by recording the predecessors of each
/// basic block and tracing a particular register value back through the chain
/// of derivations that produced it to check the property described above.  This
/// is analogous to tracing a wire connected to a circuit element back to its
/// source.
///
/// The SafeStack instrumentation pass will move any objects to the unsafe stack
/// that could have pointers to them passed between functions.  This is
/// necessary, since the X86FixupSeparateStack pass only performs
/// intra-procedural analysis.
///
/// Even with the SafeStack pass enabled, va_list objects containing pointers to
/// the stack may still be passed between C/C++ functions.  That is why the
/// Clang frontend recognizes when this pass is enabled and adds the appropriate
/// segment override prefixes to __builtin_va_arg invocations.
///
class X86FixupSeparateStack : public MachineFunctionPass {
public:
  static char ID;

  X86FixupSeparateStack();

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  llvm::StringRef getPassName() const override {
    return FIXUPSEPSTK_DESC;
  }

  const TargetInstrInfo *TII;
  const X86RegisterInfo *TRI;
  const X86Subtarget *STI;
  unsigned FrameReg;
  unsigned StackReg;

  /// Map from basic block to corresponding AddrRegReqs and StackPtrSpillReqs
  /// objects
  std::map<MachineBasicBlock *,
           std::pair<std::shared_ptr<AddrRegReqs>,
                     std::shared_ptr<StackPtrSpillReqs> > > Reqs;

  /// \brief Perform fixups for a string instruction.
  ///
  /// \param I Iterator pointing to the string instruction in the basic block.
  ///          This routine may add new instructions around the string
  ///          instruction, in which case 'I' will then point to the instruction
  ///          that was added at the highest address.
  /// \param StringDS true if the instruction accesses DS
  /// \param StringES true if the instruction accesses ES
  ///
  /// \returns true if the basic block was changed
  bool processStringInstr(MachineBasicBlock *BB, MachineBasicBlock::iterator &I,
                          AddrRegReqs &AReqs, bool StringDS, bool StringES);

  /// \brief Perform fixups for an instruction that accesses memory with an
  /// explicit memory operand.
  ///
  /// \param MemoryOperand Offset to the memory operand within the instruction
  ///
  /// \returns true if the instruction was changed
  bool processMemAccessInstr(MachineInstr &I, int MemoryOperand,
                             AddrRegReqs &AReqs);

  enum FrameOp {
    NoFrameOp, // or an irrelevant frame operation
    SpillReg,
    FillRegWithStackPtr
  };

  /// \brief Check whether an instruction is a register spill or fill and, if
  /// so, update SpilledStackPtrs appropriately.
  ///
  /// \param PossibleSP Set to true if this instruction may possibly spill or
  ///                   fill a stack pointer.  Some instructions spill and fill
  ///                   registers, but are assumed to not handle stack pointers
  ///                   (e.g. FP instructions).
  /// \param StackOffset The offset from the stack pointer value at the
  ///                    beginning of the function to the current stack pointer
  ///                    value.
  ///
  /// \returns type of frame operation performed by the instruction
  FrameOp checkForRegisterSpill(MachineInstr &I, int MemoryOperand,
                                bool PossibleSP,
                                AddrRegReqs &AReqs, StackPtrSpillReqs &SReqs,
                                int64_t StackOffset);

  /// \brief Check whether an instruction is of a type that may be used to
  /// compute a derived pointer.
  ///
  /// \returns empty set if the instruction is not of a type that may be used
  /// to compute a derived pointer.  Otherwise, returns the registers that may
  /// be used as the sources of the possible derivation.
  DenseSet<unsigned> mayComputePtr(const MachineInstr &I) const;

  /// \brief Process a basic block.
  ///
  /// \param StackOffset The offset from the stack pointer value at the
  ///                    beginning of the function to the current stack pointer
  ///                    value.
  ///
  /// \returns true if the basic block was changed
  bool processBasicBlock(MachineBasicBlock *BB,
                         int64_t StackOffset,
                         std::shared_ptr<AddrRegReqs> AReqs,
                         std::shared_ptr<StackPtrSpillReqs> SReqs);
};

char X86FixupSeparateStack::ID = 0;
}

INITIALIZE_PASS(X86FixupSeparateStack, FIXUPSEPSTK_NAME, FIXUPSEPSTK_DESC, false, false)

FunctionPass *llvm::createX86FixupSeparateStack() {
  return new X86FixupSeparateStack();
}

void AddrRegs::setRegType(unsigned Reg, RegValueType Type) {
  for (auto &E : RegTypes)
    if (TRI->regsOverlap(E.first, Reg)) {
      E.second = Type;
      return;
    }

  RegTypes.insert(std::make_pair(Reg, Type));
}

RegValueType AddrRegs::getRegType(unsigned Reg) const {
  assert(Register::isPhysicalRegister(Reg));

  if (TRI->regsOverlap(TRI->getStackRegister(), Reg))
    // The stack register is assumed to always point to the stack
    return RegValueType::StackPointer;

  for (auto E : RegTypes)
    if (TRI->regsOverlap(E.first, Reg))
      return E.second;

  return RegValueType::NotStackPointer;
}

AddrRegReqs::AddrRegReqs(AddrRegReqs *Predecessor) :
  TRI(Predecessor->TRI),
  OrigRegs(Predecessor->extractRegs()) {

  Predecessors.insert(Predecessor);
}

AddrRegs AddrRegReqs::extractRegs() {
  AddrRegs Regs(OrigRegs);

  // Handle the explicit derivations:
  for (auto &D : Derivatives)
    // Note that this derivation overrides the corresponding implicit one:
    Regs.setRegType(D.first, getRegType(D.first));

  return Regs;
}

void AddrRegReqs::addPredecessor(AddrRegReqs *Predecessor) {
  assert(Predecessors.insert(Predecessor).second);

  // It is correct to copy the current value of UsedInMemOp and iterate through
  // that in this method, even if this object is part of a cycle of predecessors
  // and the useInMemOp routine adds a register to this->UsedInMemOp.  In that
  // case, the useInMemOp routine will itself invoke Predecessor->useInMemOp
  // with the newly-added register, since Predecessor was added to Predecessors
  // prior to invoking useInMemOp below.
  DenseSet<unsigned> OrigUsedInMemOp(UsedInMemOp);

  for (auto N : OrigUsedInMemOp)
    Predecessor->useInMemOp(N);
}

DenseSet<unsigned> AddrRegReqs::lookupRoots(unsigned To) const {
  DenseSet<unsigned> Roots;

  assert(Register::isPhysicalRegister(To));

  for (auto &D : Derivatives)
    if (TRI->regsOverlap(D.first, To))
      return D.second;

  Roots.insert(To);

  return Roots;
}

void AddrRegReqs::derive(unsigned From, unsigned To) {
  if (TRI->regsOverlap(To, TRI->getStackRegister()))
    // The stack register is always assumed to point to the stack, regardless
    // of what instructions are used to update it.
    return;

  if (TRI->regsOverlap(To, X86::EFLAGS) ||
      TRI->regsOverlap(From, X86::EFLAGS))
    // Stack pointer values do not flow through the flags register.
    return;

  // Lookup the root register(s) from which From is derived, since B in each
  // pair (a, B) in Derivatives indicates the register values at the beginning
  // of the basic block (root register values) from which the value of a is
  // derived.
  DenseSet<unsigned> RootsFrom = lookupRoots(From);

  for (auto &D : Derivatives)
    if (TRI->regsOverlap(D.first, To)) {
      D.second.insert(RootsFrom.begin(), RootsFrom.end());
      return;
    }

  Derivatives.insert(std::make_pair(To, RootsFrom));
}

void AddrRegReqs::markNotDerived(unsigned To) {
  if (TRI->regsOverlap(To, TRI->getStackRegister()))
    return;

  for (auto &D : Derivatives)
    if (TRI->regsOverlap(D.first, To)) {
      D.second.clear();
      return;
    }

  Derivatives.insert(std::make_pair(To, DenseSet<unsigned>()));
}

RegValueType AddrRegReqs::useInMemOp(unsigned Reg) {
  // For registers that are set by instructions of the types used by the
  // compiler to compute pointers, this pass checks that each such register
  // either always points to stack data or never points to stack data at each
  // instruction that uses the register in a memory operand.
  //
  // For memory models that use the same base address for DS, ES, and SS and
  // that simply set a lower limit on DS and ES than on SS, ambiguous memory
  // operands could be handled by directing the memory operands to SS.  However,
  // this increases the exposure of the safe stacks.  Furthermore, it is not
  // applicable to memory models that use different base addresses for DS, ES,
  // and SS.
  RegValueType RegType = getRegType(Reg);
  assert(RegType != RegValueType::Ambiguous &&
         "One or more registers may point to both stack and data locations at "
         "one or more instructions that access memory, which is not supported "
         "by the X86FixupSeparateStack pass.  This ambiguity is due to at "
         "least one register used in a memory operand being defined directly "
         "or indirectly by an instruction that has multiple register inputs "
         "with different types.");

  DenseSet<unsigned> Roots = lookupRoots(Reg);

  // Roots -= UsedInMemOp (with checks for overlapping registers)
  DenseSet<unsigned> OrigRoots(Roots);
  for (auto R : OrigRoots)
    for (auto N : UsedInMemOp)
      if (TRI->regsOverlap(R, N))
        Roots.erase(R);

  // UsedInMemOp += Roots
  UsedInMemOp.insert(Roots.begin(), Roots.end());

  for (AddrRegReqs *P : Predecessors)
    for (auto R : Roots)
      assert(P->useInMemOp(R) == RegType &&
             "One or more registers may point to both stack and data "
             "locations at one or more instructions that access memory, which "
             "is not supported by the X86FixupSeparateStack pass.  This "
             "ambiguity is due to a single basic block having multiple "
             "predecessors that pass per-register values of different types "
             "into this basic block.");

  return RegType;
}

RegValueType AddrRegReqs::getRegType(unsigned Reg) const {
  RegValueType RegType = RegValueType::NotStackPointer;
  bool First = true;
  DenseSet<unsigned> From = lookupRoots(Reg);
  for (auto F : From)
    if (First) {
      RegType = OrigRegs.getRegType(F);
      First = false;
    } else if (RegType != OrigRegs.getRegType(F))
      return RegValueType::Ambiguous;

  return RegType;
}

StackPtrSpillReqs::StackPtrSpillReqs(StackPtrSpillReqs *Predecessor) :
    Demand(IntervalMapAlloc), Supply(IntervalMapAlloc), Top(false) {
  Predecessors.insert(Predecessor);
}

class LockingSPToggle {
private:
  bool Set;
  bool Value;

public:
  LockingSPToggle() : Set(false) {}

  void set(bool V) {
    if (Set) {
      assert(V == Value &&
             "Register fill would get a mixture of stack pointer and other "
             "data");
      return;
    }

    Value = V;
    Set = true;
  }

  bool get() {
    assert(Set);

    return Value;
  }
};

IntervalMap<int64_t, bool>::Allocator StackPtrSpillReqs::IntervalMapAlloc;

StackPtrSpillReqs::DemandResponse
StackPtrSpillReqs::demand(int64_t Start, uint64_t Size,
                          SmallPtrSet<StackPtrSpillReqs *, 16>& Demanders) {
  if (!Demanders.insert(this).second)
    // A cycle was detected, and the demand is not satisfied within this cycle.
    return DemandResponse::Unsatisfied;

  LockingSPToggle SP;

  int64_t End = Start + Size - 1;

  while (Start <= End) {
    int64_t NextSupply = End + 1;
    auto SupplyI = Supply.find(Start);
    if (SupplyI.valid()) {
      NextSupply = SupplyI.start();
      if (NextSupply <= Start) {
        // This part of the demand is satisfied by supply earlier in this
        // basic block.
        SP.set(*SupplyI);
        Start = SupplyI.stop() + 1;
        continue;
      }
    }

    int64_t NextDemand = End + 1;
    auto DemandI = Demand.find(Start);
    if (DemandI.valid()) {
      NextDemand = DemandI.start();
      if (NextDemand <= Start) {
        // This part of the demand was previously submitted and the result is
        // cached.
        SP.set(*DemandI);
        Start = DemandI.stop() + 1;
        continue;
      }
    }

    int64_t NextStart = std::min({ NextSupply, NextDemand, End + 1 });

    if (Top) {
      // This assumes that arguments and uninitialized frame slots are not stack
      // pointers
      SP.set(false);
      Start = NextStart;
      continue;
    }

    // Propagate this demand to the previously-visited predecessors of this
    // function.
    bool PredSP;
    bool DemandSatisfied = false;
    for (auto Pred : Predecessors) {
      auto Resp = Pred->demand(Start, NextStart - Start, Demanders);
      if (Resp == DemandResponse::Unsatisfied)
        // The demand was not satisfied by this cycle, but it may be satisfied
        // by a different predecessor.
        continue;

      bool SPLocal = Resp == DemandResponse::SatisfiedSP;
      if (!DemandSatisfied)
        PredSP = SPLocal;
      else
        assert(PredSP == SPLocal ||
               "Inconsistent register spills detected");
      DemandSatisfied = true;
    }
    if (!DemandSatisfied)
      return DemandResponse::Unsatisfied;

    // Cache the result of propagating this demand to the function's predecessors.
    Demand.insert(Start, NextStart - 1, PredSP);
    Start = NextStart;
    SP.set(PredSP);
  }

  Demanders.erase(this);

  return SP.get()? DemandResponse::SatisfiedSP : DemandResponse::SatisfiedNonSP;
}

bool StackPtrSpillReqs::isSPSpill(int64_t Start, uint64_t Size)
{
  SmallPtrSet<StackPtrSpillReqs *, 16> Demanders;
  DemandResponse Resp = demand(Start, Size, Demanders);

  assert(Resp != DemandResponse::Unsatisfied);

  return Resp == DemandResponse::SatisfiedSP;
}

void StackPtrSpillReqs::addPredecessor(StackPtrSpillReqs *Predecessor) {
  assert(Predecessors.find(Predecessor) == Predecessors.end());

  SmallPtrSet<StackPtrSpillReqs *, 16> Demanders;

  bool Satisfied = Demand.empty();
  for (auto I = Demand.begin(); I.valid(); ++I) {
    auto Resp = Predecessor->demand(I.start(), (I.stop() - I.start()) + 1,
                                    Demanders);

    // The Unsatisfied response can only be returned from a cycle that does not
    // include the function entrypoint.  There will be at least one other
    // predecessor that will be checked and that will lead back to the
    // function entrypoint, because the basic block traversal starts at
    // the function entrypoint.
    if (Resp == DemandResponse::Unsatisfied)
      continue;

    assert(Resp ==
        (*I?
            DemandResponse::SatisfiedSP :
            DemandResponse::SatisfiedNonSP));

    Satisfied = true;
  }

  assert(Satisfied);

  Predecessors.insert(Predecessor);
}

void StackPtrSpillReqs::supply(bool SP, int64_t Start, uint64_t Size) {
  int64_t End = Start + Size - 1;

  // Find the first interval, if any, that stops on or beyond Start.
  auto I = Supply.find(Start);

  if (I.valid() && I.start() < Start) {
    // The new interval overlaps with this existing interval that starts on or
    // before the start of the new interval, so the existing interval needs to
    // be shortened.
    int64_t ExistingEnd = I.stop();
    bool ExistingVal = *I;
    I.setStop(Start - 1);
    ++I;
    if (End < ExistingEnd)
      // The existing interval extends beyond the end of the new interval.
      // After this insertion, the iterator will point to the inserted
      // interval.
      I.insert(End + 1, ExistingEnd, ExistingVal);
  }

  while (I.valid() && I.stop() <= End)
    // This existing interval is completely overlapped by the new interval.
    I.erase();

  if (I.valid() && I.start() <= End)
    // This existing interval overlaps with the end of the new interval, so it
    // needs to be shortened.
    I.setStart(End + 1);

  // Insert the new interval.
  I.insert(Start, End, SP);
}

X86FixupSeparateStack::X86FixupSeparateStack() : MachineFunctionPass(ID) {
  initializeX86FixupSeparateStackPass(*PassRegistry::getPassRegistry());
}

bool X86FixupSeparateStack::processStringInstr(MachineBasicBlock *BB,
                                               MachineBasicBlock::iterator &I,
                                               AddrRegReqs &AReqs,
                                               bool StringDS, bool StringES) {
  MachineFunction &MF = *BB->getParent();

  // FIXME: Check for a segment override prefix on the string instruction and
  // skip the following if one is detected.

  bool StackDS = false;
  bool StackES = false;

  if (StringDS) {
    StackDS = AReqs.useInMemOp(X86::ESI) == RegValueType::StackPointer;
  }

  if (StringES) {
    StackES = AReqs.useInMemOp(X86::EDI) == RegValueType::StackPointer;
  }

  if (!(StackDS || StackES))
    return false;

  // Emit instructions to save a segment register, overwrite it with the
  // contents of SS, and restore its original value after the string instruction
  // is complete.
  auto tempOverwriteSegWithSS = [&](unsigned PushOp, unsigned PopOp) {
    // Save the original value of the segment register to be overwritten.
    MachineInstr *NewMI =
      MF.CreateMachineInstr(TII->get(PushOp), I->getDebugLoc());
    I = BB->insert(I, NewMI);
    // Push the stack segment selector onto the stack.
    NewMI = MF.CreateMachineInstr(TII->get(X86::PUSHSS32), I->getDebugLoc());
    I = BB->insertAfter(I, NewMI);
    // Overwrite the segment register with the stack segment selector.
    NewMI = MF.CreateMachineInstr(TII->get(PopOp), I->getDebugLoc());
    I = BB->insertAfter(I, NewMI);
    // Advance to the string instruction.
    I++;
    // Restore the original value of the segment register.
    NewMI = MF.CreateMachineInstr(TII->get(PopOp), I->getDebugLoc());
    I = BB->insertAfter(I, NewMI);
  };

  if (StackDS) {
    tempOverwriteSegWithSS(X86::PUSHDS32, X86::POPDS32);
  }
  if (StackES) {
    if (StackDS)
      // Back up to the string instruction.
      I--;
    tempOverwriteSegWithSS(X86::PUSHES32, X86::POPES32);
    if (StackDS)
      // Advance to the POP instruction that restores DS.
      I++;
  }

  return true;
}

bool X86FixupSeparateStack::processMemAccessInstr(MachineInstr &I,
                                                  int MemoryOperand,
                                                  AddrRegReqs &AReqs) {
  MachineOperand &SegRegOp = I.getOperand(MemoryOperand + X86::AddrSegmentReg);

  unsigned PrevSegReg = SegRegOp.getReg();

  if (Register::isPhysicalRegister(PrevSegReg))
    // Do not replace the existing segment override prefix.
    return false;

  unsigned BaseReg = I.getOperand(MemoryOperand + X86::AddrBaseReg).getReg();

  bool BaseIsPhysReg = Register::isPhysicalRegister(BaseReg);

  bool InStack = false;

  if (BaseIsPhysReg)
    InStack = AReqs.useInMemOp(BaseReg) == RegValueType::StackPointer;

  // if (InStack &&
  //      (TRI->regsOverlap(X86::ESP, BaseReg) ||
  //       TRI->regsOverlap(X86::EBP, BaseReg)))
  //     // Memory operand with a base register of ESP or EBP implicitly references
  //     // SS.
  //   return false;

  // if (!InStack &&
  //     // Memory operand without a base register implicitly
  //     // references DS.
  //     (!(BaseIsPhysReg &&
  //        // Memory operand with a base register other than ESP and EBP implicitly
  //        // references DS.  This assumes that ESP is never used as a non-stack
  //        // pointer.
  //        TRI->regsOverlap(X86::EBP, BaseReg))))
  //   return false;

  if (InStack) {
    SegRegOp.ChangeToRegister(X86::UR, false);
  }

  return true;
}

DenseSet<unsigned> X86FixupSeparateStack::mayComputePtr(const MachineInstr &I) const {
  DenseSet<unsigned> PossibleBases;

  if (I.isCall())
    // It is possible that a function may return a pointer value, but we assume
    // that functions never return stack pointer values.  This is based on the
    // assumption that stack pointers are never passed between functions when
    // the separate stack feature is enabled.
    //
    // Returning the empty set here omits the possible derivation of a non-stack
    // pointer value from some caller register that is input to the callee.
    // However, such a derivation cannot result in a violation of the
    // correctness requirements that are checked by this pass.  Specifically,
    // the consistency requirements could only be violated if the return value
    // of the function is ambiguous, i.e. may be both a stack pointer value in
    // some control flows and and some other type of value in other control
    // flows.  As explained above, this would violate the assumptions of this
    // pass.
    return PossibleBases;

  bool HasDef = false;
  for (const MachineOperand &Op : I.operands()) {
    if (!Op.isReg())
      continue;

    unsigned Reg = Op.getReg();

    if (!Register::isPhysicalRegister(Reg))
      continue;

    if (Op.isDef())
      HasDef = true;

    if (!Op.isUse() || Op.isUndef() || Op.isImplicit())
      continue;

    PossibleBases.insert(Reg);

    // This assumes that LEA32r is used to compute an offset from a base
    // address, and that the base address is specified in the first register
    // use operand.  Thus, later register operands are not recorded as
    // possible base registers.
    //
    // FIXME: Support additional instructions that perform complex
    // arithmetic on stack pointer values.
    if (I.getOpcode() == X86::LEA32r)
      break;
  }

  if (!HasDef)
    PossibleBases.clear();

  return PossibleBases;
}

X86FixupSeparateStack::FrameOp
X86FixupSeparateStack::checkForRegisterSpill(MachineInstr &I,
                                             int MemoryOperand,
                                             bool PossibleSP,
                                             AddrRegReqs &AReqs,
                                             StackPtrSpillReqs &SReqs,
                                             int64_t StackOffset) {
  unsigned BaseReg = I.getOperand(MemoryOperand + X86::AddrBaseReg).getReg();

  if (!Register::isPhysicalRegister(BaseReg))
    return NoFrameOp;

  if (!TRI->regsOverlap(FrameReg, BaseReg))
    return NoFrameOp;

  const MachineOperand &Disp = I.getOperand(MemoryOperand + X86::AddrDisp);

  // FIXME: Consider whether other forms of addressing should be handled.
  if (!Disp.isImm())
    return NoFrameOp;

  bool FoundMemOp = false;
  uint64_t Size;
  for (auto MO : I.memoperands()) {
    if (FoundMemOp)
      // This assumes that register spill/fill instructions have only a single
      // memory operand.
      return NoFrameOp;

    FoundMemOp = true;
    Size = MO->getSize();
  }

  if (!FoundMemOp)
    return NoFrameOp;

  int64_t disp = Disp.getImm();
  if (BaseReg == StackReg)
    disp += StackOffset;

  if (I.mayStore()) {
    RegValueType SrcType = RegValueType::NotStackPointer;
    if (PossibleSP) {
      int RegOpNo = I.mayStore()? MemoryOperand + X86::AddrNumOperands : 0;
      SrcType = AReqs.getRegType(I.getOperand(RegOpNo).getReg());
    }

    assert(SrcType != RegValueType::Ambiguous &&
           "Spill of ambiguous register data not supported by "
           "X86FixupSeparateStack pass.");

    SReqs.supply(SrcType == RegValueType::StackPointer, disp, Size);
    return SpillReg;
  }

  if (SReqs.isSPSpill(disp, Size))
    return FillRegWithStackPtr;

  return NoFrameOp;
}

bool X86FixupSeparateStack::processBasicBlock(MachineBasicBlock *BB,
                                              int64_t StackOffset,
                                              std::shared_ptr<AddrRegReqs> AReqs,
                                              std::shared_ptr<StackPtrSpillReqs> SReqs) {
  bool Changed = false;

  Reqs.emplace(BB, std::make_pair(AReqs, SReqs));

  for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    bool StringDS, StringES;

    unsigned Opcode = I->getOpcode();

    switch (Opcode) {
    case X86::REP_MOVSB_32:
    case X86::REP_MOVSW_32:
    case X86::REP_MOVSD_32:
      StringDS = true;
      StringES = true;
      break;
    case X86::REP_STOSB_32:
    case X86::REP_STOSW_32:
    case X86::REP_STOSD_32:
      StringDS = false;
      StringES = true;
      break;
    default:
      StringDS = false;
      StringES = false;
      break;
    }

    if (StringDS || StringES) {
      Changed |= processStringInstr(BB, I, *AReqs, StringDS, StringES);
      continue;
    }

    const MCInstrDesc &Desc = TII->get(Opcode);
    uint64_t TSFlags = Desc.TSFlags;

    int MemoryOperand = -1;
    if (I->mayLoadOrStore()) {
      // Determine where the memory operand starts, if present.
      MemoryOperand = X86II::getMemoryOperandNo(TSFlags);
      if (MemoryOperand != -1)
        MemoryOperand += X86II::getOperandBias(Desc);
    }

    bool HasEVEX_K = TSFlags & X86II::EVEX_K;
    bool HasVEX_4V = TSFlags & X86II::VEX_4V;

    DenseSet<unsigned> MayDerivePtrFrom;

    // This assumes that the compiler only uses certain forms of instructions
    // to spill and fill registers.  Certain VEX-encoded instructions are
    // specifically excluded to reduce the complexity of the subsequent code
    // that computes the operand number of the register being stored.  See
    // X86MCCodeEmitter::EmitVEXOpcodePrefix for information about how these
    // instructions are represented in LLVM.  Again, this assumes that such
    // instructions are not used to spill and fill registers.
    if (I->mayLoadOrStore() && !I->isCall() && MemoryOperand != -1) {
      bool PossibleSP =
          (((TSFlags & X86II::FormMask) == X86II::MRMDestMem) ||
           ((TSFlags & X86II::FormMask) == X86II::MRMSrcMem)) &&
          !(HasEVEX_K || HasVEX_4V);

      FrameOp FrmOp = NoFrameOp;
      FrmOp = checkForRegisterSpill(*I, MemoryOperand, PossibleSP, *AReqs, *SReqs, StackOffset);

      if (FrmOp == FillRegWithStackPtr) {
        assert(PossibleSP);
        MayDerivePtrFrom.insert(StackReg);
      }
    }

    // This pass is unable to track stack pointers that are written to memory
    // other than spilled register values.  Thus, the SafeStack pass should be
    // used with this pass to prevent (safe) stack pointers from being written
    // to memory if those pointers could subsequently be incorrectly used as
    // pointers into a segment other than SS.
    //
    // As an example of a stack pointer store that is allowable, consider C
    // variadic argument handling.  va_start and va_arg may write stack
    // pointers to memory, but Clang uses the appropriate address space number
    // to access variadic arguments in the stack segment.

    if (MemoryOperand != -1)
      Changed |= processMemAccessInstr(*I, MemoryOperand, *AReqs);

    // This assumes that tracking the stack growth is unnecessary when a
    // dedicated frame register is used.
    if (FrameReg == StackReg &&
        I->definesRegister(StackReg)) {
      // This assumes that only the following instructions are used to update ESP.
      switch (Opcode) {
      case X86::ADD32ri:
      case X86::ADD32ri8:
        assert(I->getOperand(0).getReg() == StackReg &&
               I->getOperand(1).getReg() == StackReg &&
               "Unsupported instruction");
        StackOffset += I->getOperand(2).getImm();
        break;
      case X86::SUB32ri:
      case X86::SUB32ri8:
        assert(I->getOperand(0).getReg() == StackReg &&
               I->getOperand(1).getReg() == StackReg &&
               "Unsupported instruction");
        StackOffset -= I->getOperand(2).getImm();
        break;
      case X86::LEA32r:
        assert(I->getOperand(0).getReg() == StackReg &&
               I->getOperand(1).getReg() == StackReg &&
               I->getOperand(2).getImm() == 1 &&
               I->getOperand(3).getReg() == X86::NoRegister &&
               I->getOperand(5).getReg() == X86::NoRegister &&
               "Unsupported instruction");
        StackOffset += I->getOperand(4).getImm();
        break;
      case X86::PUSHi32:
      case X86::PUSH32i8:
      case X86::PUSH32r:
      case X86::PUSH32rmm:
        StackOffset -= 4;
        break;
      case X86::POP32r:
      case X86::POP32rmm:
        StackOffset += 4;
        break;
      default:
        // CALL instructions do push a return address on the stack, but that
        // address is subsequently popped by a corresponding RET instruction
        // prior to executing the instruction following the CALL.
        assert(I->isCall() && "Unsupported instruction for updating stack pointer.");
        break;
      }

      assert(StackOffset <= 0 &&
             "Error occurred while computing stack offset");
    } else if (!I->mayLoadOrStore())
      MayDerivePtrFrom = mayComputePtr(*I);

    if (I->isCall()) {
      AReqs->markNotDerived(X86::EAX);
      continue;
    }

    for (MachineOperand &Op : I->operands()) {
      if (!Op.isReg() || !Op.isDef() || Op.isImplicit())
        continue;

      unsigned Reg = Op.getReg();

      if (!Register::isPhysicalRegister(Reg))
        continue;

      if (MayDerivePtrFrom.find(Reg) == MayDerivePtrFrom.end())
        AReqs->markNotDerived(Reg);

      for (unsigned N : MayDerivePtrFrom)
        AReqs->derive(N, Reg);
    }
  }

  for (auto Succ : BB->successors()) {
    auto RI = Reqs.find(Succ);
    if (RI == Reqs.end()) {
      std::shared_ptr<AddrRegReqs> SuccAReqs(new AddrRegReqs(AReqs.get()));
      std::shared_ptr<StackPtrSpillReqs> SuccSReqs(
          new StackPtrSpillReqs(SReqs.get()));
      Changed |= processBasicBlock(Succ, StackOffset, SuccAReqs, SuccSReqs);
    } else {
      RI->second.first->addPredecessor(AReqs.get());
      RI->second.second->addPredecessor(SReqs.get());
    }
  }

  return Changed;
}

bool X86FixupSeparateStack::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<X86Subtarget>();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  FrameReg = TRI->getFrameRegister(MF);
  StackReg = TRI->getStackRegister();

  if (!STI->useSeparateStackSeg())
    return false;

  if (std::find(FlatMemFuncs.begin(), FlatMemFuncs.end(), MF.getName())
      != FlatMemFuncs.end())
    return false;

  // assert(!STI->is64Bit() && "Only X86-32 is supported.");

  Reqs.clear();

  std::shared_ptr<AddrRegReqs> InitAReqs(new AddrRegReqs(TRI));
  std::shared_ptr<StackPtrSpillReqs> InitSReqs(new StackPtrSpillReqs());
  return processBasicBlock(&MF.front(), 0, InitAReqs, InitSReqs);
}