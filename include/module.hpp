#pragma once

#include "tensor.hpp"

class Module
{
  public:
    virtual const Tensor& forward(const Tensor& input) = 0;
    virtual const Tensor& backward(const Tensor& dLdZ) = 0;
    virtual void          update(float learning_rate)  = 0;
    virtual ~Module() = default; // all tensor destructors will be called automatically
};

class Linear : public Module
{

  private:
    int m, n, b; // input, output and batch dimensions

    Tensor W;  // shape (m, n)
    Tensor W0; // shape (n, 1) - bias

    Tensor dW;  // shape (m, n)
    Tensor dW0; // shape (n, 1) - bias gradient

    Tensor A; // shape (m, b) - cached input in current pass
    Tensor Z; // shape (n, b) - cached output in current pass

    Tensor dLdA; // shape (m, b) - gradient w.r.t. input (for backward pass)

    cudaStream_t stream;

  public:
    Linear(int m, int n, int b, cudaStream_t stream = 0);
    const Tensor& forward(const Tensor& input) override;
    const Tensor& backward(const Tensor& grad_output) override;
    void          update(float learning_rate) override;
};

class ReLU : public Module
{
  private:
    int          n, b; // output and batch dimensions
    Tensor       A;    // shape (n, b) - cached output in current pass
    Tensor       dLdZ; // shape (n, b) - gradient w.r.t. input (for backward pass)
    Tensor       mask; // shape (n, b) - mask for ReLU activation
    cudaStream_t stream;

  public:
    ReLU(int n, int b, cudaStream_t stream = 0);
    const Tensor& forward(const Tensor& Z) override;
    const Tensor& backward(const Tensor& dLdA) override;
    void          update(float learning_rate) override {} // ReLU has no parameters to update
};

class SoftMax : public Module
{
  private:
    int          n, b; // output and batch dimensions
    Tensor       A;    // shape (n, b) - cached output in current pass
    Tensor       dLdZ; // shape (n, b) - gradient w.r.t. input (for backward pass)
    cudaStream_t stream;

  public:
    SoftMax(int n, int b, cudaStream_t stream = 0);
    const Tensor& forward(const Tensor& Z) override;
    const Tensor& backward(const Tensor& dLdA) override;
    void          update(float learning_rate) override {} // SoftMax has no parameters to update
};

class CrossEntropyLoss
{

  private:
    int          n;
    int          b;    // output and batch dimensions
    Tensor       dLdA; // shape (n, b) - gradient w.r.t. input (for backward pass)
    cudaStream_t stream;

  public:
    CrossEntropyLoss(int n, int b, cudaStream_t stream = 0);

    float         calc_loss(const Tensor& y_pred, const Tensor& y_true);
    const Tensor& calc_grad(const Tensor& y_pred, const Tensor& y_true);
    Tensor        predict_class(const Tensor& y_pred);
};
