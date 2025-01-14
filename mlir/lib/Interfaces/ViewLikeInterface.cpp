//===- ViewLikeInterface.cpp - View-like operations in MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ViewLike Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the loop-like interfaces.
#include "mlir/Interfaces/ViewLikeInterface.cpp.inc"

LogicalResult mlir::verifyListOfOperandsOrIntegers(Operation *op,
                                                   StringRef name,
                                                   unsigned numElements,
                                                   ArrayRef<int64_t> staticVals,
                                                   ValueRange values) {
  // Check static and dynamic offsets/sizes/strides does not overflow type.
  if (staticVals.size() != numElements)
    return op->emitError("expected ") << numElements << " " << name
                                      << " values, got " << staticVals.size();
  unsigned expectedNumDynamicEntries =
      llvm::count_if(staticVals, [](int64_t staticVal) {
        return ShapedType::isDynamic(staticVal);
      });
  if (values.size() != expectedNumDynamicEntries)
    return op->emitError("expected ")
           << expectedNumDynamicEntries << " dynamic " << name << " values";
  return success();
}

LogicalResult
mlir::detail::verifyOffsetSizeAndStrideOp(OffsetSizeAndStrideOpInterface op) {
  std::array<unsigned, 3> maxRanks = op.getArrayAttrMaxRanks();
  // Offsets can come in 2 flavors:
  //   1. Either single entry (when maxRanks == 1).
  //   2. Or as an array whose rank must match that of the mixed sizes.
  // So that the result type is well-formed.
  if (!(op.getMixedOffsets().size() == 1 && maxRanks[0] == 1) && // NOLINT
      op.getMixedOffsets().size() != op.getMixedSizes().size())
    return op->emitError(
               "expected mixed offsets rank to match mixed sizes rank (")
           << op.getMixedOffsets().size() << " vs " << op.getMixedSizes().size()
           << ") so the rank of the result type is well-formed.";
  // Ranks of mixed sizes and strides must always match so the result type is
  // well-formed.
  if (op.getMixedSizes().size() != op.getMixedStrides().size())
    return op->emitError(
               "expected mixed sizes rank to match mixed strides rank (")
           << op.getMixedSizes().size() << " vs " << op.getMixedStrides().size()
           << ") so the rank of the result type is well-formed.";

  if (failed(verifyListOfOperandsOrIntegers(
          op, "offset", maxRanks[0], op.getStaticOffsets(), op.getOffsets())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "size", maxRanks[1], op.getStaticSizes(), op.getSizes())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "stride", maxRanks[2], op.getStaticStrides(), op.getStrides())))
    return failure();

  for (int64_t offset : op.getStaticOffsets()) {
    if (offset < 0 && !ShapedType::isDynamic(offset))
      return op->emitError("expected offsets to be non-negative, but got ")
             << offset;
  }
  for (int64_t size : op.getStaticSizes()) {
    if (size < 0 && !ShapedType::isDynamic(size))
      return op->emitError("expected sizes to be non-negative, but got ")
             << size;
  }
  return success();
}

static char getLeftDelimiter(AsmParser::Delimiter delimiter) {
  switch (delimiter) {
  case AsmParser::Delimiter::Paren:
    return '(';
  case AsmParser::Delimiter::LessGreater:
    return '<';
  case AsmParser::Delimiter::Square:
    return '[';
  case AsmParser::Delimiter::Braces:
    return '{';
  default:
    llvm_unreachable("unsupported delimiter");
  }
}

static char getRightDelimiter(AsmParser::Delimiter delimiter) {
  switch (delimiter) {
  case AsmParser::Delimiter::Paren:
    return ')';
  case AsmParser::Delimiter::LessGreater:
    return '>';
  case AsmParser::Delimiter::Square:
    return ']';
  case AsmParser::Delimiter::Braces:
    return '}';
  default:
    llvm_unreachable("unsupported delimiter");
  }
}

enum class ListTypeMode {
  // No type is parsed/printed either for values or integers.
  NoType,
  // The type of each value is parsed/printed next to its name. No type is
  // parsed/printed for integers.
  PerElement,
  // The type for all the values and integers is parsed/printed once at the end
  // of the list.
  SameType
};

static void printDynamicIndexListImpl(OpAsmPrinter &printer, Operation *op,
                                      OperandRange values,
                                      ArrayRef<int64_t> integers,
                                      ArrayRef<bool> scalables,
                                      ListTypeMode typePrintMode,
                                      AsmParser::Delimiter delimiter,
                                      Type sameElementType = Type()) {
  char leftDelimiter = getLeftDelimiter(delimiter);
  char rightDelimiter = getRightDelimiter(delimiter);
  printer << leftDelimiter;
  if (integers.empty()) {
    printer << rightDelimiter;
    return;
  }

  unsigned dynamicValIdx = 0;
  unsigned scalableIndexIdx = 0;
  llvm::interleaveComma(integers, printer, [&](int64_t integer) {
    if (!scalables.empty() && scalables[scalableIndexIdx])
      printer << "[";
    if (ShapedType::isDynamic(integer)) {
      printer << values[dynamicValIdx];
      if (typePrintMode == ListTypeMode::PerElement)
        printer << " : " << values[dynamicValIdx].getType();
      ++dynamicValIdx;
    } else {
      printer << integer;
    }
    if (!scalables.empty() && scalables[scalableIndexIdx])
      printer << "]";

    scalableIndexIdx++;
  });

  // Print the common type for all the list elements.
  if (typePrintMode == ListTypeMode::SameType) {
    assert(sameElementType && "expected a valid type");

    if (!values.empty() || !sameElementType.isInteger(64))
      printer << " : " << sameElementType;
  }

  printer << rightDelimiter;
}

void mlir::printDynamicIndexList(OpAsmPrinter &printer, Operation *op,
                                 OperandRange values,
                                 ArrayRef<int64_t> integers,
                                 ArrayRef<bool> scalables,
                                 TypeRange valueTypes,
                                 AsmParser::Delimiter delimiter) {
  assert((valueTypes.empty() || values.size() == valueTypes.size()) &&
         "The number of values and value types mismatch");
  assert((valueTypes.empty() || llvm::equal(values.getTypes(), valueTypes)) &&
         "The type of values and value types mismatch");
  ListTypeMode typePrintMode =
      valueTypes.empty() ? ListTypeMode::NoType : ListTypeMode::PerElement;

  printDynamicIndexListImpl(printer, op, values, integers, scalables,
                            typePrintMode, delimiter);
}

void mlir::printSameTypeDynamicIndexList(OpAsmPrinter &printer, Operation *op,
                                         OperandRange values,
                                         DenseIntElementsAttr integers,
                                         TypeRange valueTypes,
                                         AsmParser::Delimiter delimiter) {
  bool hasElements = !values.empty() || !integers.empty();
  ListTypeMode typePrintMode =
      hasElements ? ListTypeMode::SameType : ListTypeMode::NoType;

  Type sameElementType;
  if (hasElements) {
    sameElementType =
        values.empty() ? integers.getElementType() : values.front().getType();

    assert(integers.getElementType() == sameElementType &&
           "Expected the same type for integers and values");
    assert(llvm::all_of(llvm::zip_equal(values.getTypes(), valueTypes),
                        [&](auto zipIt) {
                          return std::get<0>(zipIt) == sameElementType &&
                                 std::get<1>(zipIt) == sameElementType;
                        }) &&
           "Expected the same type for all the values and value types");
  }

  SmallVector<int64_t> integerElements;
  integers.getAsIntegers(integerElements);
  printDynamicIndexListImpl(printer, op, values, integerElements,
                            /*scalables=*/{}, typePrintMode, delimiter,
                            sameElementType);
}

static ParseResult parseDynamicIndexListImpl(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    SmallVectorImpl<int64_t> &integers, DenseBoolArrayAttr *scalables,
    ListTypeMode typeParseMode,
    SmallVectorImpl<Type> *elementTypes, AsmParser::Delimiter delimiter) {

  assert((typeParseMode == ListTypeMode::NoType || elementTypes) &&
         "Expected value types in per-element and same-type mode");

  SmallVector<bool, 4> scalableVals;
  auto parseIntegerOrValue = [&]() {
    OpAsmParser::UnresolvedOperand operand;
    auto res = parser.parseOptionalOperand(operand);

    // Handle values.
    if (res.has_value() && succeeded(res.value())) {
      values.push_back(operand);
      integers.push_back(ShapedType::kDynamic);
      if (typeParseMode == ListTypeMode::PerElement &&
          parser.parseColonType(elementTypes->emplace_back()))
        return failure();
    } else {
      // Handle integers.
      // When encountering `[`, assume that this is a scalable index.
      if (scalables)
        scalableVals.push_back(parser.parseOptionalLSquare().succeeded());

      int64_t integer;
      if (failed(parser.parseInteger(integer)))
        return failure();
      integers.push_back(integer);

      // If this is assumed to be a scalable index, verify that there's a
      // closing
      // `]`.
      if (scalables && scalableVals.back() &&
          parser.parseOptionalRSquare().failed())
        return failure();
    }

    if (typeParseMode == ListTypeMode::SameType) {
      Type sameType;
      // Type is optional for integers. If a value is present, the type is
      // mandatory.
      if (parser.parseOptionalColonType(sameType) && !values.empty())
        return failure();

      if (sameType)
        elementTypes->push_back(sameType);
    }

    return success();
  };

  if (parser.parseCommaSeparatedList(delimiter, parseIntegerOrValue,
                                     " in dynamic index list"))
    return parser.emitError(parser.getNameLoc())
           << "expected SSA value or integer";

  if (scalables)
    *scalables = parser.getBuilder().getDenseBoolArrayAttr(scalableVals);
  return success();
}

ParseResult mlir::parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, DenseBoolArrayAttr &scalables,
    SmallVectorImpl<Type> *valueTypes, AsmParser::Delimiter delimiter) {

  ListTypeMode typeParseMode =
      valueTypes ? ListTypeMode::PerElement : ListTypeMode::NoType;
  SmallVector<int64_t> integerElements;
  if (failed(parseDynamicIndexListImpl(parser, values, integerElements,
                                       &scalables, typeParseMode, valueTypes,
                                       delimiter)))
    return failure();

  integers = parser.getBuilder().getDenseI64ArrayAttr(integerElements);
  return success();
}

ParseResult mlir::parseSameTypeDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseIntElementsAttr &integers, SmallVectorImpl<Type> &valueTypes,
    AsmParser::Delimiter delimiter) {

  SmallVector<int64_t> integerElements;
  SmallVector<Type> elementTypes;
  if (failed(parseDynamicIndexListImpl(parser, values, integerElements,
                                       /*scalables=*/nullptr,
                                       ListTypeMode::SameType, &elementTypes,
                                       delimiter)))
    return failure();

  // If only integer elements are present but no type, default to i64.
  if (values.empty() && !integerElements.empty() && elementTypes.empty())
    elementTypes.push_back(parser.getBuilder().getI64Type());

  bool hasElements = !values.empty() || !integerElements.empty();
  if (hasElements) {
    assert(elementTypes.size() == 1 && "Expected a single type");
    integers = DenseIntElementsAttr::get(
        VectorType::get(integerElements.size(), elementTypes[0]),
        integerElements);
  }

  if (!values.empty())
    valueTypes.assign(values.size(), elementTypes[0]);

  return success();
}

bool mlir::detail::sameOffsetsSizesAndStrides(
    OffsetSizeAndStrideOpInterface a, OffsetSizeAndStrideOpInterface b,
    llvm::function_ref<bool(OpFoldResult, OpFoldResult)> cmp) {
  if (a.getStaticOffsets().size() != b.getStaticOffsets().size())
    return false;
  if (a.getStaticSizes().size() != b.getStaticSizes().size())
    return false;
  if (a.getStaticStrides().size() != b.getStaticStrides().size())
    return false;
  for (auto it : llvm::zip(a.getMixedOffsets(), b.getMixedOffsets()))
    if (!cmp(std::get<0>(it), std::get<1>(it)))
      return false;
  for (auto it : llvm::zip(a.getMixedSizes(), b.getMixedSizes()))
    if (!cmp(std::get<0>(it), std::get<1>(it)))
      return false;
  for (auto it : llvm::zip(a.getMixedStrides(), b.getMixedStrides()))
    if (!cmp(std::get<0>(it), std::get<1>(it)))
      return false;
  return true;
}

unsigned mlir::detail::getNumDynamicEntriesUpToIdx(ArrayRef<int64_t> staticVals,
                                                   unsigned idx) {
  return std::count_if(staticVals.begin(), staticVals.begin() + idx,
                       [&](int64_t val) { return ShapedType::isDynamic(val); });
}
