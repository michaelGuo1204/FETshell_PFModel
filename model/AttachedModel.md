以简化模型为例, 其中$\phi$表示区域, $m$为表示某一物质浓度的序参量, $D,d$为模型参数
$$
\frac{\part \phi m}{\part t} = D\nabla \cdot (\phi\nabla m) + \phi(a-d m)
$$
直接处理$D\nabla \cdot (\phi\nabla m)$较为复杂, 我们将其展开为
$$
\frac{\part \phi m}{\part t} = D\left(\phi\cdot \nabla^2 m + \nabla \phi \cdot \nabla m\right) + \phi(a-d m)
$$
由于在一次迭代中$\phi$固定, 左右两端同除$\phi$. 同时用时间步长$h$​差分, 使用半隐格式
$$
\frac{m^{(t+1)}-m^{(t)}}{h} = \underbrace{D\nabla^2m^{(t+1)}+(a-dm^{(t+1)}) }_{\text{隐}} + \underbrace{\frac{D}{\phi}\nabla \phi\cdot \nabla  m^{(t)}}_{\text{显}}
$$
对于二维情况, 引入波束矢量$\mathbf k$, 使用傅里叶谱方法求解​
$$
\begin{aligned}
\hat m^{(t+1)}-\hat m^{(t)} &= -Dhk^2\hat m^{(t+1)} + \hat a - dh\hat{m}^{(t+1)}+\frac{Dh}{\phi}\widehat{(\nabla \phi)}*\widehat{ \nabla  m^{(t)}}\\
(1+ Dhk^2 + dh)\hat m^{(t+1)}&= \hat m^{(t)}+\hat a+ Dh\mathcal{F}\left(\mathcal{F}^{-1}(ik_x\hat\phi / \phi)\cdot \mathcal{F}^{-1}(ik_x \hat m^{(t)})+\right.\\
&\quad  \left.\mathcal{F}^{-1}(ik_x\hat\phi / \phi)\cdot \mathcal{F}^{-1}(ik_x \hat m^{(t)})\right)
\end{aligned}
$$
因此有
$$
\hat m^{(t+1)}= \left(\hat m^{(t)}+\hat a+ Dh\mathcal{F}\left(\mathcal{F}^{-1}(ik_x\hat\phi / \phi)\cdot \mathcal{F}^{-1}(ik_x \hat m^{(t)})+\mathcal{F}^{-1}(ik_y\hat\phi / \phi)\cdot \mathcal{F}^{-1}(ik_y \hat m^{(t)})\right)\right)/(1+ Dhk^2 + dh)
$$