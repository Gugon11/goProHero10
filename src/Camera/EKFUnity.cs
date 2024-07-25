using System;
using UnityEngine;

public class EKF
{
    private float dt;
    private float a;
    private float b;
    private Matrix4x4 Q;
    private Matrix4x4 R;
    private Vector3 X;
    private Matrix4x4 P;

    public EKF(float dt, float a, float b, Matrix4x4 Q, Matrix4x4 R, Vector3 initialState, Matrix4x4 initialCovariance)
    {
        this.dt = dt;
        this.a = a;
        this.b = b;
        this.Q = Q;
        this.R = R;
        this.X = initialState;
        this.P = initialCovariance;
    }

    private float WrapToPi(float angle)
    {
        return Mathf.Repeat(angle + Mathf.PI, 2 * Mathf.PI) - Mathf.PI;
    }

    public void Predict(float v, float delta)
    {
        float theta = X.z;
        float alpha = Mathf.Atan(a * Mathf.Tan(delta) / b);

        X.x += v * Mathf.Cos(alpha + theta) * dt;
        X.y += v * Mathf.Sin(alpha + theta) * dt;
        X.z += (v / b) * Mathf.Tan(delta) * dt;
        X.z = WrapToPi(X.z);

        Matrix4x4 Fx = Matrix4x4.identity;
        Fx[0, 2] = -v * Mathf.Sin(alpha + theta) * dt;
        Fx[1, 2] = v * Mathf.Cos(alpha + theta) * dt;

        Matrix4x4 Fw = Matrix4x4.zero;
        Fw[0, 0] = Mathf.Cos(alpha + theta) * dt;
        Fw[1, 0] = Mathf.Sin(alpha + theta) * dt;
        Fw[2, 1] = dt;

        P = Fx * P * Fx.transpose + Fw * Q * Fw.transpose;
    }

    public void Update(Vector3 z)
    {
        Vector3 h = new Vector3(X.x, X.y, X.z);

        Matrix4x4 Hx = Matrix4x4.identity;

        Vector3 y = z - h;
        y.z = WrapToPi(y.z);

        Matrix4x4 S = Hx * P * Hx.transpose + R;
        Matrix4x4 K = P * Hx.transpose * S.inverse;

        X += new Vector3(K[0, 0] * y.x + K[0, 1] * y.y + K[0, 2] * y.z,
                         K[1, 0] * y.x + K[1, 1] * y.y + K[1, 2] * y.z,
                         K[2, 0] * y.x + K[2, 1] * y.y + K[2, 2] * y.z);

        P = (Matrix4x4.identity - K * Hx) * P;
    }

    public Vector3 GetState()
    {
        return X;
    }
}
