/**
 * @file image.h
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */
#ifndef SRC_IMAGE_H
#define SRC_IMAGE_H

#include <cstring>

namespace vio_tools
{
    template<typename T>
    class Image
    {
    public:
        Image(int rows, int cols)
                : rows_(rows),
                  cols_(cols)
        {
            data_ = new T[rows_ * cols_];
        }

        Image(int rows, int cols, T *data)
                : rows_(rows),
                  cols_(cols)
        {
            data_ = new T[rows_ * cols_];
            std::memcpy(data_, data, rows_ * cols_ * sizeof(T));
        }

        const T &operator()(int r, int c) const
        {
            int i = r;
            int j = c;

            if (i < 0)
                i = 0;
            else if (i >= rows_)
                i = rows_ - 1;

            if (j < 0)
                i = 0;
            else if (j >= cols_)
                j = cols_ - 1;

            return data_[i * cols_ + j];
        }

        T &operator()(int r, int c)
        {
            int i = r;
            int j = c;

            if (i < 0)
                i = 0;
            else if (i >= rows_)
                i = rows_ - 1;

            if (j < 0)
                i = 0;
            else if (j >= cols_)
                j = cols_ - 1;

            return data_[i * cols_ + j];
        }

        ~Image()
        {
            if (data_ != nullptr)
                delete[] data_;
        }

        void upload(const T *const data)
        {
            std::memcpy(data_, data, rows_ * cols_ * sizeof(T));
        }

        void download(T *data)
        {
            std::memcpy(data, data_, rows_ * cols_ * sizeof(T));
        }

        int rows() const
        {
            return rows_;
        }

        int rows()
        {
            return rows_;
        }

        int cols() const
        {
            return cols_;
        }

        int cols()
        {
            return cols_;
        }

        T *data()
        {
            return data_;
        }

        const T *data() const
        {
            return data_;
        }

    private:
        int rows_, cols_;

        T *data_;
    };
}
#endif //SRC_IMAGE_H
