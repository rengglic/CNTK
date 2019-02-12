#include "stdafx.h"
#include "TopKMatrix.h"
#include "ColumnQuantizer.h"
#include "Constants.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
TopKMatrix<ElemType>::TopKMatrix(const size_t numRows, const size_t numCols, const int topK, DEVICEID_TYPE deviceId, MemAllocator* allocator /* = nullptr */)
    : m_numRows(numRows), m_numCols(numCols), m_topK(topK), m_allocator(allocator)
{
    m_qColSize = TopKColumn<ElemType>::TopKColumnSize(topK);
    size_t realNumCols = GetNumColumns();

    if (m_allocator == nullptr)
    {
        m_topKData = new Matrix<char>(1, m_qColSize * realNumCols, deviceId);
    }
    else
    {
        m_topKData = new Matrix<char>(1, m_qColSize * realNumCols, (char*)m_allocator->Malloc(m_qColSize * realNumCols), deviceId, matrixFlagDontOwnBuffer);
    }
}

template <class ElemType>
TopKMatrix<ElemType>::TopKMatrix(TopKMatrix<ElemType>&& moveFrom)
    : m_topKData(moveFrom.m_topKData), m_allocator(moveFrom.m_allocator), m_numRows(moveFrom.m_numRows), m_numCols(moveFrom.m_numCols),  m_qColSize(moveFrom.m_qColSize)
{
    moveFrom.m_topKData = nullptr;
    moveFrom.m_allocator = nullptr;
}

template <class ElemType>
TopKMatrix<ElemType>& TopKMatrix<ElemType>::operator=(TopKMatrix<ElemType>&& moveFrom)
{
    assert(this != &moveFrom);

    this->m_topKData = moveFrom.m_topKData;
    this->m_allocator = moveFrom.m_allocator;
    this->m_numRows = moveFrom.m_numRows;
    this->m_numCols = moveFrom.m_numCols;
    this->m_topK = moveFrom.m_topK;
    this->m_qColSize = moveFrom.m_qColSize;

    moveFrom.m_topKData = nullptr;
    moveFrom.m_allocator = nullptr;

    return *this;
}

template <class ElemType>
TopKMatrix<ElemType>::TopKMatrix(const size_t numRows, const size_t numCols, const int topK, Matrix<char>* data)
    : m_numRows(numRows), m_numCols(numCols), m_topK(topK), m_topKData(data), m_allocator(nullptr)
{
    m_qColSize = TopKColumn<ElemType>::TopKColumnSize(topK);
}

template <class ElemType>
TopKMatrix<ElemType>::~TopKMatrix()
{
    if (nullptr != m_topKData)
    {
        // If we used an external allocator, lets free the backing buffer of the matrix
        if (m_allocator != nullptr)
        {
            assert(!m_topKData->OwnBuffer());
            m_allocator->Free(m_topKData->Data());
        }

        delete m_topKData;
        m_topKData = nullptr;
    }
}

template <class ElemType>
int TopKMatrix<ElemType>::GetDeviceId() const
{
    return m_topKData->GetDeviceId();
}

template <class ElemType>
size_t TopKMatrix<ElemType>::GetNumColumns() const
{
    return (m_numCols + (TOPK_NUMELEMENTS - 1)) / TOPK_NUMELEMENTS;
}

template <class ElemType>
size_t TopKMatrix<ElemType>::GetSize() const
{
    return m_topKData->GetNumElements();
}

template <class ElemType>
char* TopKMatrix<ElemType>::Buffer() const
{
    return m_topKData->Data();
}

template <class ElemType>
TopKMatrix<ElemType> TopKMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    size_t dataStartColumn = startColumn / TOPK_NUMELEMENTS * m_qColSize;
    size_t dataNumCols = (numCols + (TOPK_NUMELEMENTS - 1)) / TOPK_NUMELEMENTS * m_qColSize;
    auto matrixSliceData = new Matrix<char>(m_topKData->ColumnSlice(dataStartColumn, dataNumCols));

    return TopKMatrix<ElemType>(this->GetNumRows(), numCols, this->GetTopK(), matrixSliceData);
}

template <class ElemType>
void TopKMatrix<ElemType>::Print(const char* matrixName, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd)
{
    if ((GetNumRows() == 0) || (GetNumCols() == 0))
    {
        LogicError("Print: TopKMatrix is empty.");
    }

    // if (rowEnd >= GetNumRows() || colEnd >= GetNumCols())
    // {
    //     InvalidArgument("Index out of range.");
    // }

    // DEVICEID_TYPE orgdevice = this->GetDeviceId();
    // CurrentDataLocation curLocation = m_topKData->GetCurrentMatrixLocation();
    // if (curLocation == CurrentDataLocation::GPU)
    // {
    //     m_topKData->_transferToDevice(CPUDEVICE, false, false);
    // }

    // if (matrixName != nullptr)
    //     fprintf(stderr, "\n###### %s (%lu, %lu) ######\n", matrixName, (unsigned long)GetNumRows(), (unsigned long)GetNumCols());
    // else
    //     fprintf(stderr, "\n###### Unnamed Matrix (%lu, %lu) ######\n", (unsigned long)GetNumRows(), (unsigned long)GetNumCols());

    // fprintf(stderr, "\n------ Print Range (%lu:%lu, %lu:%lu) ------\n", (unsigned long)rowStart, (unsigned long)rowEnd, (unsigned long)colStart, (unsigned long)colEnd);

    // for (size_t j = colStart; j <= colEnd; j++)
    // {
    //     TopKColumn<ElemType>* qCol = this->GetTopKColumn(j);
    //     fprintf(stderr, "Scaling factor=%.10f\t", qCol->scalingFactor);
    // }
    // fprintf(stderr, "\n");

    // const size_t ldNbits = ValueQuantizer<ElemType>::ld(this->GetNumBits());
    // size_t numQWordsPerCol = ColumnQuantizer<ElemType>::QWordsPerCol(TOPK_NUMELEMENTS, this->GetNumBits());
    // for (size_t i = rowStart; i <= rowEnd; i++)
    // {
    //     size_t qWordIdx = i % numQWordsPerCol;
    //     size_t offsetInQWord = i / numQWordsPerCol;
    //     for (size_t j = colStart; j <= colEnd; j++)
    //     {
    //         TopKColumn<ElemType>* qCol = this->GetTopKColumn(j);
    //         ColumnQuantizer<ElemType> q(ldNbits, qCol->scalingFactor);
    //         QWord qWord = qCol->bits[qWordIdx];

    //         QWordVal qVal;
    //         ElemType val;
            
    //         const QWordVal bitmask = q.valQ.QuanRangeEnd() - 1;
    //         qVal = (qWord >> (offsetInQWord * this->GetNumBits())) & bitmask;
    //         val = q.valQ.Unquantize(qVal);

    //         if (qCol->scalingFactor > 1e-12)
    //             fprintf(stderr, "%10d (%.10f) %10lf         \t", (int) qVal, val, (double) qCol->scalingFactor);
    //     }
    //     fprintf(stderr, "\n");
    // }

    // if (curLocation == CurrentDataLocation::GPU)
    // {
    //     m_topKData->_transferToDevice(orgdevice, false, false);
    // }
}

// Explicit instantiation
template class TopKMatrix<float>;
template class TopKMatrix<double>;

}}}
