import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiFileText, FiMessageSquare, FiUpload, FiShield, FiZap, FiLayers } from 'react-icons/fi';
import { FaGoogle } from 'react-icons/fa';

export default function Landing() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.15 }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
  };

  const features = [
    {
      icon: FiUpload,
      title: 'Document Upload',
      description: 'Upload PDFs, EPUBs, DOCX, and more. Our system processes and indexes your documents for intelligent retrieval.',
      color: 'from-neon-blue to-neon-purple'
    },
    {
      icon: FiMessageSquare,
      title: 'AI-Powered Queries',
      description: 'Ask questions in natural language. Get precise answers sourced directly from your uploaded documents.',
      color: 'from-neon-purple to-neon-pink'
    },
    {
      icon: FiLayers,
      title: 'Smart Retrieval',
      description: 'Advanced RAG pipeline with semantic search and generative AI for accurate, context-aware responses.',
      color: 'from-neon-pink to-neon-green'
    },
    {
      icon: FiShield,
      title: 'Secure Authentication',
      description: 'Email/password and Google OAuth support. Your documents and queries are protected with enterprise-grade security.',
      color: 'from-neon-green to-neon-blue'
    }
  ];

  const steps = [
    { step: '1', title: 'Sign Up', description: 'Create an account with email or Google OAuth' },
    { step: '2', title: 'Upload', description: 'Upload your documents in any supported format' },
    { step: '3', title: 'Query', description: 'Ask questions and get AI-powered answers' },
  ];

  return (
    <div className="min-h-screen bg-dark-950 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-neon-blue/10 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-neon-purple/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-neon-pink/5 rounded-full blur-3xl animate-float" style={{ animationDelay: '4s' }} />
      </div>

      {/* Grid Pattern Overlay */}
      <div
        className="absolute inset-0 opacity-5"
        style={{
          backgroundImage: 'linear-gradient(rgba(0, 243, 255, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 243, 255, 0.1) 1px, transparent 1px)',
          backgroundSize: '50px 50px'
        }}
      />

      {/* Content */}
      <div className="relative z-10">
        {/* Navigation Bar */}
        <motion.nav
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-blue to-neon-purple flex items-center justify-center">
              <FiZap className="text-white text-xl" />
            </div>
            <span className="text-2xl font-bold gradient-text">N.O.V.A.R</span>
          </div>
          <div className="flex items-center gap-4">
            <Link to="/login" className="text-white/80 hover:text-white font-medium transition-colors">
              Sign In
            </Link>
            <Link to="/signup" className="btn-primary">
              Get Started
            </Link>
          </div>
        </motion.nav>

        {/* Hero Section */}
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="max-w-5xl mx-auto px-8 py-20 text-center"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.3, type: 'spring' }}
            className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full mb-8 text-sm text-neon-blue"
          >
            <FiZap />
            <span>Powered by Gemini AI & RAG Technology</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="text-5xl md:text-7xl font-bold mb-6"
          >
            <span className="gradient-text">Intelligent Document</span>
            <br />
            <span className="text-white">Understanding at Your Fingertips</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="text-xl text-white/60 max-w-3xl mx-auto mb-10"
          >
            Upload your documents, ask questions in natural language, and get AI-powered answers
            sourced directly from your content. Built with cutting-edge Retrieval-Augmented Generation.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link to="/signup" className="btn-primary text-lg px-8 py-4">
              Start Free
            </Link>
            <Link to="/login" className="btn-secondary text-lg px-8 py-4 flex items-center gap-2">
              <FaGoogle className="text-xl" />
              <span>Continue with Google</span>
            </Link>
          </motion.div>
        </motion.section>

        {/* Features Section */}
        <motion.section
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
          className="max-w-7xl mx-auto px-8 py-20"
        >
          <motion.h2
            variants={item}
            className="text-4xl font-bold text-center mb-4 gradient-text"
          >
            Powerful Features
          </motion.h2>
          <motion.p
            variants={item}
            className="text-white/60 text-center mb-16 text-lg"
          >
            Everything you need to unlock insights from your documents
          </motion.p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                variants={item}
                className="glass rounded-2xl p-8 neon-border hover:border-neon-blue/50 transition-all duration-300 group"
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <feature.icon className="text-white text-2xl" />
                </div>
                <h3 className="text-2xl font-bold mb-3 text-white">{feature.title}</h3>
                <p className="text-white/60 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* How It Works */}
        <motion.section
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
          className="max-w-5xl mx-auto px-8 py-20"
        >
          <motion.h2
            variants={item}
            className="text-4xl font-bold text-center mb-4 gradient-text"
          >
            How It Works
          </motion.h2>
          <motion.p
            variants={item}
            className="text-white/60 text-center mb-16 text-lg"
          >
            Get started in three simple steps
          </motion.p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((step, index) => (
              <motion.div
                key={step.step}
                variants={item}
                className="text-center relative"
              >
                <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-neon-blue to-neon-purple flex items-center justify-center text-2xl font-bold">
                  {step.step}
                </div>
                <h3 className="text-xl font-bold mb-3">{step.title}</h3>
                <p className="text-white/60">{step.description}</p>
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-8 left-2/3 w-1/3 border-t-2 border-dashed border-white/20" />
                )}
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* CTA Section */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto px-8 py-20 text-center"
        >
          <div className="glass rounded-3xl p-12 neon-border">
            <h2 className="text-4xl font-bold gradient-text mb-4">Ready to Get Started?</h2>
            <p className="text-white/60 text-lg mb-8">
              Sign up today and start unlocking the insights hidden in your documents.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/signup" className="btn-primary text-lg px-8 py-4">
                Create Free Account
              </Link>
              <Link to="/login" className="text-white/80 hover:text-white font-medium transition-colors text-lg">
                Already have an account? Sign In
              </Link>
            </div>
          </div>
        </motion.section>

        {/* Footer */}
        <footer className="border-t border-white/10 py-8 text-center text-white/40 text-sm">
          <p>N.O.V.A.R &mdash; Network of Vectorized Archive Retrieval</p>
        </footer>
      </div>
    </div>
  );
}
