# EAPG-for-MOP
This is an algorithm for solving nonsmooth multiobjective objective problems,even for nonconvex,
which uses a smoothing function to establish a subproblem,equivalenting to the original problem.
main2.m is just a try,I will update it soon.
in this code,I use a test problem as follow:
assume there is a multifunction
\tilde{F}(x,\mu) = \tilde{f}(x,\mu)+g(x)，
where
g(x) = (0.01 \left\|x\right\|_1,0.01 \left\|x\right\|_1),

\tilde{f}(x,\mu) = (\sum_{i=1}^{m} \tilde{\theta} (A_i x -b_i,\mu) ,\sum_{i=1}^{m} \tilde{\theta} (\tilde{\phi}(A_i x -b_i),\mu) )	
where
\begin{align*}
	\tilde{\theta}(s,\mu)=
	\begin{cases}
		|s| \ if |s| > \mu, \\
		\frac{s^2}{2 \mu} + \frac{\mu}{2} \ if |s| \leq \mu
	\end{cases}
\end{align*}

\begin{align*}
	\tilde{\phi}(s,\mu)=
	\begin{cases}
		\max \{s,0\} \ if |s| > \mu, \\
		\frac{(s+\mu)^2}{4 \mu} \ if |s| \leq \mu
	\end{cases}
\end{align*}

