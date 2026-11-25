/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// Symbol visibility for the aiehlc dialects' symbol ops. Since LLVM 24 every
// SymbolOpInterface op must provide getVisibility/setVisibility (upstream
// 979b72237338); the upstream SymbolVisibility trait needs a `sym_visibility`
// inherent attribute these ops never had. Every aiehlc symbol is public.
#ifndef AIEHLC_SYMBOL_H
#define AIEHLC_SYMBOL_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace aiehlc {
template <typename ConcreteType> class PublicSymbol : public ::mlir::OpTrait::TraitBase<ConcreteType, PublicSymbol> {
  public:
    ::mlir::SymbolTable::Visibility getVisibility() { return ::mlir::SymbolTable::Visibility::Public; }
    void setVisibility(::mlir::SymbolTable::Visibility) {}
};

/* Ops that return a value cannot be symbols since LLVM 24 ("symbols must not
 * have results", upstream ff7896e0fa2e), so dmaphop.port / consumer / producer,
 * dmap.define_port_configure, dfschedule.packet, ... keep their `sym_name` but
 * are found by name: the first OpT named like ref's leaf, searching from the
 * nearest symbol-table ancestor of `from` outward (lookupNearestSymbolFrom's
 * order). */
template <typename OpT> OpT lookupNamed(::mlir::Operation *from, ::mlir::SymbolRefAttr ref) {
    if (!from || !ref)
        return {};
    const ::llvm::StringRef name = ref.getLeafReference().getValue();
    for (::mlir::Operation *scope = from; scope; scope = scope->getParentOp()) {
        if (!scope->hasTrait<::mlir::OpTrait::SymbolTable>())
            continue;
        OpT found;
        scope->walk([&](OpT op) {
            auto n = op->template getAttrOfType<::mlir::StringAttr>("sym_name");
            if (n && n.getValue() == name) {
                found = op;
                return ::mlir::WalkResult::interrupt();
            }
            return ::mlir::WalkResult::advance();
        });
        if (found)
            return found;
    }
    return {};
}
} // namespace aiehlc

#endif // AIEHLC_SYMBOL_H
