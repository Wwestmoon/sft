#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据生成器包
"""

from .main import SFTDataGenerator
from .base import LLMAPI, OpenAICompatibleAPI, AnswerExtractor, SimpleAnswerExtractor, TableConverter, ResultSaver
from .decomposition import QuestionDecomposer
from .sub_question_solver import SubQuestionSolver
from .answer_synthesizer import AnswerSynthesizer
from .quality_control import QualityController, TestRunner
from .refine_critic import CriticRefine, ErrorAnalyzer, ErrorFixer
from .concurrency import ConcurrentProcessor, RateLimiter, AsyncProcessor

__version__ = '1.0.0'
__author__ = 'Your Name'
__description__ = 'SFT 训练数据生成器'
