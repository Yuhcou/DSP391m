# Documentation Index

Complete guide to DSP2 documentation.

## 📖 Documentation Overview

This documentation suite covers all aspects of the DSP2 multi-elevator EGCS system, from quick setup to deep technical details.

---

## 🚀 Getting Started

### [README.md](../README.md)
**Start here!** Project overview, quick start, and feature highlights.

**Key Sections:**
- Installation instructions
- Quick start (5 minutes)
- System architecture overview
- Performance benchmarks
- Configuration examples

**Best For:**
- First-time users
- Project overview
- Quick reference

### [QUICKSTART.md](QUICKSTART.md)
**Hands-on tutorial** to get training in 5 minutes.

**Key Sections:**
- Installation & setup
- First training run (3 options: quick, simple, full)
- Evaluating performance
- Understanding results
- Troubleshooting common issues
- Next steps and workflows

**Best For:**
- New users wanting to train immediately
- Learning by doing
- Quick testing and debugging
- Common workflow patterns

---

## 🏗️ Technical Documentation

### [ARCHITECTURE.md](ARCHITECTURE.md)
**Comprehensive system design** documentation.

**Key Sections:**
- System architecture overview
- Environment design (EGCSEnv, PassengerTracker, AdaptiveRewardCalculator)
- Agent implementation (DDQN, networks, replay buffer)
- Traditional algorithms (5 baseline dispatchers)
- Training & evaluation pipelines
- Configuration system
- Design patterns & best practices

**Best For:**
- Understanding system internals
- Contributing to the codebase
- Research and analysis
- Advanced customization
- Performance optimization

**Topics Covered:**
- State/action space design
- Reward structure details
- Network architectures
- Multi-agent coordination
- Curriculum learning
- Action masking
- VDN mixing

### [API_REFERENCE.md](API_REFERENCE.md)
**Complete API documentation** for all classes and functions.

**Key Sections:**
- Environment API (EGCSEnv, PassengerTracker, AdaptiveRewardCalculator)
- Agent API (DDQNAgent, DQNNet, ReplayBuffer)
- Traditional Algorithms API (all 5 dispatchers)
- Utilities API (masking, simulation helpers)
- Type hints and error handling
- Code examples

**Best For:**
- Implementing custom features
- Understanding function signatures
- Looking up parameter meanings
- Code examples
- Integration with other systems

**Includes:**
- Full function signatures
- Parameter descriptions
- Return value specifications
- Usage examples
- Error handling patterns

### [REWARD_MODEL.md](REWARD_MODEL.md)
**Deep dive into reward engineering** and tuning.

**Key Sections:**
- Base reward structure (penalties, shaping)
- Adaptive reward system (5 strategies)
- Reward engineering techniques (capping, clipping, scaling)
- Comprehensive tuning guide
- Debugging reward issues
- Detailed examples

**Best For:**
- Tuning reward functions
- Understanding training dynamics
- Fixing learning issues
- Advanced reward design
- Research on reward shaping

**Topics Covered:**
- Penalty capping rationale
- Reward shaping benefits
- Adaptive reward strategies
- Dynamic weight adjustment
- Curriculum design
- Common pitfalls
- Tuning workflows

---

## 📚 Quick Reference Guide

### By User Type

#### 🆕 **First-Time Users**
1. Start: [README.md](../README.md) - Overview
2. Next: [QUICKSTART.md](QUICKSTART.md) - Get running
3. Then: [REWARD_MODEL.md](REWARD_MODEL.md) - Understand training

#### 👨‍💻 **Developers**
1. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
2. [API_REFERENCE.md](API_REFERENCE.md) - Function APIs
3. [REWARD_MODEL.md](REWARD_MODEL.md) - Reward tuning

#### 🔬 **Researchers**
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
2. [REWARD_MODEL.md](REWARD_MODEL.md) - Reward engineering
3. [API_REFERENCE.md](API_REFERENCE.md) - Implementation

#### 🐛 **Debugging Issues**
1. [QUICKSTART.md](QUICKSTART.md#troubleshooting) - Common issues
2. [REWARD_MODEL.md](REWARD_MODEL.md#debugging-reward-issues) - Reward problems
3. [ARCHITECTURE.md](ARCHITECTURE.md#troubleshooting) - System issues

### By Task

#### **Setting Up**
- Installation → [README.md](../README.md#-quick-start)
- First run → [QUICKSTART.md](QUICKSTART.md#first-training-run)
- Configuration → [ARCHITECTURE.md](ARCHITECTURE.md#5-configuration-system)

#### **Training**
- Basic training → [QUICKSTART.md](QUICKSTART.md#first-training-run)
- Hyperparameters → [ARCHITECTURE.md](ARCHITECTURE.md#hyperparameters)
- Adaptive rewards → [REWARD_MODEL.md](REWARD_MODEL.md#adaptive-reward-system)

#### **Evaluation**
- Quick eval → [QUICKSTART.md](QUICKSTART.md#evaluating-performance)
- Baselines → [README.md](../README.md#-training--evaluation)
- Metrics → [QUICKSTART.md](QUICKSTART.md#understanding-results)

#### **Customization**
- New environment → [API_REFERENCE.md](API_REFERENCE.md#environment-api)
- Custom agent → [API_REFERENCE.md](API_REFERENCE.md#agent-api)
- New rewards → [REWARD_MODEL.md](REWARD_MODEL.md#reward-engineering-techniques)

#### **Debugging**
- Not learning → [QUICKSTART.md](QUICKSTART.md#troubleshooting)
- Reward issues → [REWARD_MODEL.md](REWARD_MODEL.md#debugging-reward-issues)
- Instability → [ARCHITECTURE.md](ARCHITECTURE.md#troubleshooting)

---

## 📋 Document Comparison

| Document | Pages | Level | Best For |
|----------|-------|-------|----------|
| README.md | 1 | Beginner | Overview, quick start |
| QUICKSTART.md | 5-6 | Beginner | Tutorial, hands-on |
| ARCHITECTURE.md | 15-20 | Advanced | System design, internals |
| API_REFERENCE.md | 20-25 | Advanced | API details, integration |
| REWARD_MODEL.md | 15-20 | Advanced | Reward engineering |

---

## 🎯 Learning Paths

### Path 1: Quick Start (30 minutes)
1. [README.md](../README.md) - 5 min read
2. [QUICKSTART.md](QUICKSTART.md) - 10 min tutorial
3. Run training - 15 min
4. Review results

**Outcome:** Trained agent, basic understanding

### Path 2: Deep Understanding (2-3 hours)
1. [README.md](../README.md) - Overview
2. [QUICKSTART.md](QUICKSTART.md) - Hands-on
3. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
4. [REWARD_MODEL.md](REWARD_MODEL.md) - Reward tuning

**Outcome:** Comprehensive understanding, ready to customize

### Path 3: API Integration (1-2 hours)
1. [README.md](../README.md) - Context
2. [API_REFERENCE.md](API_REFERENCE.md) - API details
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Design patterns
4. Code integration

**Outcome:** Successfully integrated into your project

### Path 4: Research (4-6 hours)
1. [README.md](../README.md) - Background
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system
3. [REWARD_MODEL.md](REWARD_MODEL.md) - Reward design
4. [API_REFERENCE.md](API_REFERENCE.md) - Implementation
5. Experiment design

**Outcome:** Ready for research experiments

---

## 🔍 Search by Topic

### Environment
- **Overview**: [ARCHITECTURE.md § Environment](ARCHITECTURE.md#1-environment-dsp2env)
- **API**: [API_REFERENCE.md § Environment](API_REFERENCE.md#environment-api)
- **State/Action Space**: [README.md § State & Action Space](../README.md#state--action-space)

### Agent
- **Overview**: [ARCHITECTURE.md § Agent](ARCHITECTURE.md#2-agent-dsp2agents)
- **API**: [API_REFERENCE.md § Agent](API_REFERENCE.md#agent-api)
- **Training**: [QUICKSTART.md § Training](QUICKSTART.md#first-training-run)

### Rewards
- **Base Rewards**: [REWARD_MODEL.md § Base Reward](REWARD_MODEL.md#base-reward-structure)
- **Adaptive Rewards**: [REWARD_MODEL.md § Adaptive](REWARD_MODEL.md#adaptive-reward-system)
- **Tuning**: [REWARD_MODEL.md § Tuning](REWARD_MODEL.md#tuning-guide)

### Traditional Algorithms
- **Overview**: [ARCHITECTURE.md § Traditional Algorithms](ARCHITECTURE.md#3-traditional-algorithms-dsp2agentstraditional_algorithmspy)
- **API**: [API_REFERENCE.md § Traditional](API_REFERENCE.md#traditional-algorithms-api)
- **Evaluation**: [QUICKSTART.md § Baselines](QUICKSTART.md#2-compare-against-baselines)

### Configuration
- **System**: [ARCHITECTURE.md § Configuration](ARCHITECTURE.md#5-configuration-system)
- **Examples**: [README.md § Configuration](../README.md#-configuration)
- **Tuning**: [REWARD_MODEL.md § Tuning](REWARD_MODEL.md#tuning-guide)

### Troubleshooting
- **Quick fixes**: [QUICKSTART.md § Troubleshooting](QUICKSTART.md#troubleshooting)
- **Reward issues**: [REWARD_MODEL.md § Debugging](REWARD_MODEL.md#debugging-reward-issues)
- **System issues**: [ARCHITECTURE.md § Troubleshooting](ARCHITECTURE.md#troubleshooting)

---

## 📊 Code Examples

### Basic Usage
- [QUICKSTART.md § Example Usage](QUICKSTART.md#-example-usage)
- [README.md § Example Usage](../README.md#-example-usage)

### API Examples
- [API_REFERENCE.md § Examples](API_REFERENCE.md#examples)

### Advanced Patterns
- [ARCHITECTURE.md § Design Patterns](ARCHITECTURE.md#design-patterns--best-practices)

### Reward Examples
- [REWARD_MODEL.md § Examples](REWARD_MODEL.md#examples)

---

## 🔗 External Resources

### Research Papers
- [DDQN](https://arxiv.org/abs/1509.06461) - Double Q-learning
- [Dueling DQN](https://arxiv.org/abs/1511.06581) - Value decomposition
- [VDN](https://arxiv.org/abs/1706.05296) - Multi-agent value decomposition
- [PER](https://arxiv.org/abs/1511.05952) - Prioritized experience replay

### Related Work
- Elevator dispatching surveys
- Multi-agent reinforcement learning
- Reward shaping techniques
- Curriculum learning

---

## 📝 Contributing to Documentation

Found an error or want to improve documentation?

**What to improve:**
- Typos or unclear explanations
- Missing examples
- Outdated information
- Additional use cases

**Where to contribute:**
- [README.md](../README.md) - Overview & getting started
- [QUICKSTART.md](QUICKSTART.md) - Tutorials & troubleshooting
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [REWARD_MODEL.md](REWARD_MODEL.md) - Reward engineering

---

## 🎓 Glossary

Quick reference for common terms:

| Term | Definition | Doc Reference |
|------|------------|---------------|
| **EGCS** | Elevator Group Control System | [README](../README.md) |
| **DDQN** | Double Deep Q-Network | [ARCHITECTURE](ARCHITECTURE.md) |
| **AWT** | Average Waiting Time | [QUICKSTART](QUICKSTART.md) |
| **AJT** | Average Journey Time | [QUICKSTART](QUICKSTART.md) |
| **VDN** | Value Decomposition Networks | [ARCHITECTURE](ARCHITECTURE.md) |
| **PER** | Prioritized Experience Replay | [API_REFERENCE](API_REFERENCE.md) |
| **Action Masking** | Preventing invalid actions | [ARCHITECTURE](ARCHITECTURE.md) |
| **Reward Shaping** | Adding helper rewards | [REWARD_MODEL](REWARD_MODEL.md) |
| **Curriculum Learning** | Progressive difficulty | [REWARD_MODEL](REWARD_MODEL.md) |

---

## 💡 Tips for Navigation

### Finding Information
1. **Known topic?** Use [Search by Topic](#-search-by-topic)
2. **New to project?** Follow [Learning Path 1](#path-1-quick-start-30-minutes)
3. **Need API details?** Go to [API_REFERENCE.md](API_REFERENCE.md)
4. **Fixing issues?** Check [Troubleshooting](#by-task)

### Using the Docs
- Each document has **Table of Contents** at top
- Use **Ctrl+F** to search within documents
- Follow **cross-references** between documents
- Check **code examples** for practical usage

### Getting Help
1. Check [QUICKSTART.md Troubleshooting](QUICKSTART.md#troubleshooting)
2. Review [REWARD_MODEL.md Debugging](REWARD_MODEL.md#debugging-reward-issues)
3. Search documentation for error message
4. Open GitHub issue with details

---

## 📅 Last Updated

This documentation index was created to help users navigate the DSP2 documentation suite. Documentation is maintained alongside code changes.

**Documentation Suite:**
- [README.md](../README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Tutorial guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical reference
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation  
- [REWARD_MODEL.md](REWARD_MODEL.md) - Reward engineering
- [INDEX.md](INDEX.md) - This document

---

**Happy learning! 📚**
