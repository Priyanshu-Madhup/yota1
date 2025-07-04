import React, { useState } from 'react';
import './Login_new.css';

function Login({ onLogin, onRegister }) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isLoading) return;

    setError('');
    setSuccess('');

    // Basic validation
    if (!formData.username || !formData.password) {
      setError('Please fill in all required fields');
      return;
    }

    if (!isLogin) {
      if (!formData.email) {
        setError('Please enter your email');
        return;
      }
      if (formData.password !== formData.confirmPassword) {
        setError('Passwords do not match');
        return;
      }
      if (formData.password.length < 6) {
        setError('Password must be at least 6 characters long');
        return;
      }
    }

    setIsLoading(true);

    try {
      if (isLogin) {
        // Login
        const result = await onLogin({
          username: formData.username,
          password: formData.password
        });
        
        if (!result.success) {
          setError(result.error || 'Login failed. Please check your credentials.');
        } else {
          setSuccess('Login successful! Redirecting...');
        }
      } else {
        // Register
        const result = await onRegister({
          username: formData.username,
          email: formData.email,
          password: formData.password
        });
        
        if (!result.success) {
          setError(result.error || 'Registration failed. Please try again.');
        } else {
          setSuccess('Registration successful! You are now logged in.');
        }
      }
    } catch (error) {
      console.error('Authentication error:', error);
      setError('Authentication failed. Please check your connection and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setFormData({
      username: '',
      email: '',
      password: '',
      confirmPassword: ''
    });
    setError('');
    setSuccess('');
  };

  return (
    <div className="login-container">
      {/* Floating orbs for visual appeal */}
      <div className="orb-1"></div>
      <div className="orb-2"></div>
      <div className="orb-3"></div>
      
      <div className="login-background">
        <div className="login-form-container">
          {/* Header with Yota Logo */}
          <div className="login-header">
            <img src="./yota_logo.png" alt="Yota Logo" className="yota-logo" />
            <h1 className="login-title">Yota AI</h1>
            <p className="login-subtitle">
              {isLogin ? 'Welcome back to the future of AI conversation' : 'Join the next generation of AI assistance'}
            </p>
          </div>

          {/* Error/Success Messages */}
          {error && <div className="error-message">{error}</div>}
          {success && <div className="success-message">{success}</div>}

          {/* Login/Signup Form */}
          <form onSubmit={handleSubmit} className="login-form">
            {/* Username Field */}
            <div className="form-group">
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleInputChange}
                placeholder="Username"
                className="form-input"
                required
                disabled={isLoading}
              />
            </div>

            {/* Email Field (only for signup) */}
            {!isLogin && (
              <div className="form-group">
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  placeholder="Email Address"
                  className="form-input"
                  required
                  disabled={isLoading}
                />
              </div>
            )}

            {/* Password Field */}
            <div className="form-group">
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="Password"
                className="form-input"
                required
                disabled={isLoading}
              />
            </div>

            {/* Confirm Password Field (only for signup) */}
            {!isLogin && (
              <div className="form-group">
                <input
                  type="password"
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  placeholder="Confirm Password"
                  className="form-input"
                  required
                  disabled={isLoading}
                />
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              className={`login-button ${isLoading ? 'loading' : ''}`}
              disabled={isLoading}
            >
              {isLoading ? (
                <span>Processing...</span>
              ) : (
                isLogin ? 'Sign In' : 'Create Account'
              )}
            </button>
          </form>

          {/* Toggle between Login/Register */}
          <div className="auth-toggle">
            <p>
              {isLogin ? "Don't have an account?" : "Already have an account?"}
            </p>
            <button
              type="button"
              onClick={toggleMode}
              className="toggle-button"
              disabled={isLoading}
            >
              {isLogin ? 'Create Account' : 'Sign In'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;
