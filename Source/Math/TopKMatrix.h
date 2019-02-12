#pragma once

#include "Matrix.h"
#include "MemAllocator.h"
#include "ValueQuantizer.h"

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define MATH_API
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// A TopKColumn represents a topK value of a column as a byte stream of this format:
template <class ElemType>
struct TopKColumn
{
public:
    unsigned long long int valuesAndPositions[1];

    cudasharedcode static size_t TopKColumnSize(size_t rows)
    {
        const size_t columnDataSize = rows * sizeof(unsigned long long int);
        return columnDataSize;
    }
};

class MATH_API TopKMatrixBase
{};

template <class ElemType>
class MATH_API TopKMatrix : public TopKMatrixBase
{
public:
    TopKMatrix(const size_t numRows, const size_t numCols, const int topK, DEVICEID_TYPE deviceId, MemAllocator* allocator = nullptr);

    // Move constructor and assignment
    TopKMatrix(TopKMatrix<ElemType>&& moveFrom);
    TopKMatrix<ElemType>& operator=(TopKMatrix<ElemType>&& moveFrom);

    ~TopKMatrix();

    size_t GetNumColumns() const;

    int GetDeviceId() const;

    size_t GetNumRows() const
    {
        return m_numRows;
    }

    size_t GetNumCols() const
    {
        return m_numCols;
    }

    int GetTopK() const
    {
        return m_topK;
    }

    size_t GetSize() const;
    char* Buffer() const;

    TopKColumn<ElemType>* GetTopKColumn(size_t colIdx)
    {
        return (TopKColumn<ElemType>*) (&((this->Buffer())[m_qColSize * colIdx]));
    }

    Matrix<char>* GetTopKData() const
    {
        return m_topKData;
    }

    TopKMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;

    void Print(const char* matrixName, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd);

private:
    // Private constructor for creating topK matrix column slices
    TopKMatrix(const size_t numRows, const size_t numCols, const int topK, Matrix<char>* data);

    // Disallow copy construction and assignment
    TopKMatrix(const TopKMatrix<ElemType>&) = delete;
    TopKMatrix<ElemType>& operator=(const TopKMatrix<ElemType>&) = delete;

private:
    Matrix<char>* m_topKData;
    MemAllocator* m_allocator;

    size_t m_numRows;
    size_t m_numCols;
    int m_topK;

    // number of bytes in a topK column
    size_t m_qColSize;

    template <typename T>
    friend class MatrixQuantizer;
};

}}}
