# Generated from gram.g4 by ANTLR 4.7.2
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\17")
        buf.write("Y\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
        buf.write("\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r\4\16")
        buf.write("\t\16\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\3\3\3\3\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3")
        buf.write("\4\3\5\3\5\3\6\3\6\3\6\3\7\3\7\3\b\3\b\3\t\3\t\3\n\3\n")
        buf.write("\3\13\3\13\3\f\6\fH\n\f\r\f\16\fI\3\r\3\r\7\rN\n\r\f\r")
        buf.write("\16\rQ\13\r\3\16\6\16T\n\16\r\16\16\16U\3\16\3\16\2\2")
        buf.write("\17\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27\r")
        buf.write("\31\16\33\17\3\2\6\3\2\62;\5\2C\\aac|\6\2\62;C\\aac|\5")
        buf.write("\2\13\f\16\17\"\"\2[\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2")
        buf.write("\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2")
        buf.write("\21\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31")
        buf.write("\3\2\2\2\2\33\3\2\2\2\3\35\3\2\2\2\5$\3\2\2\2\7,\3\2\2")
        buf.write("\2\t\67\3\2\2\2\139\3\2\2\2\r<\3\2\2\2\17>\3\2\2\2\21")
        buf.write("@\3\2\2\2\23B\3\2\2\2\25D\3\2\2\2\27G\3\2\2\2\31K\3\2")
        buf.write("\2\2\33S\3\2\2\2\35\36\7U\2\2\36\37\7v\2\2\37 \7c\2\2")
        buf.write(" !\7v\2\2!\"\7g\2\2\"#\7u\2\2#\4\3\2\2\2$%\7C\2\2%&\7")
        buf.write("e\2\2&\'\7v\2\2\'(\7k\2\2()\7q\2\2)*\7p\2\2*+\7u\2\2+")
        buf.write("\6\3\2\2\2,-\7v\2\2-.\7t\2\2./\7c\2\2/\60\7p\2\2\60\61")
        buf.write("\7u\2\2\61\62\7k\2\2\62\63\7v\2\2\63\64\7k\2\2\64\65\7")
        buf.write("q\2\2\65\66\7p\2\2\66\b\3\2\2\2\678\7<\2\28\n\3\2\2\2")
        buf.write("9:\7/\2\2:;\7@\2\2;\f\3\2\2\2<=\7=\2\2=\16\3\2\2\2>?\7")
        buf.write(".\2\2?\20\3\2\2\2@A\7-\2\2A\22\3\2\2\2BC\7]\2\2C\24\3")
        buf.write("\2\2\2DE\7_\2\2E\26\3\2\2\2FH\t\2\2\2GF\3\2\2\2HI\3\2")
        buf.write("\2\2IG\3\2\2\2IJ\3\2\2\2J\30\3\2\2\2KO\t\3\2\2LN\t\4\2")
        buf.write("\2ML\3\2\2\2NQ\3\2\2\2OM\3\2\2\2OP\3\2\2\2P\32\3\2\2\2")
        buf.write("QO\3\2\2\2RT\t\5\2\2SR\3\2\2\2TU\3\2\2\2US\3\2\2\2UV\3")
        buf.write("\2\2\2VW\3\2\2\2WX\b\16\2\2X\34\3\2\2\2\6\2IOU\3\b\2\2")
        return buf.getvalue()


class gramLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    STATES = 1
    ACTIONS = 2
    TRANSITION = 3
    DPOINT = 4
    FLECHE = 5
    SEMI = 6
    VIRG = 7
    PLUS = 8
    LCROCH = 9
    RCROCH = 10
    INT = 11
    ID = 12
    WS = 13

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'States'", "'Actions'", "'transition'", "':'", "'->'", "';'", 
            "','", "'+'", "'['", "']'" ]

    symbolicNames = [ "<INVALID>",
            "STATES", "ACTIONS", "TRANSITION", "DPOINT", "FLECHE", "SEMI", 
            "VIRG", "PLUS", "LCROCH", "RCROCH", "INT", "ID", "WS" ]

    ruleNames = [ "STATES", "ACTIONS", "TRANSITION", "DPOINT", "FLECHE", 
                  "SEMI", "VIRG", "PLUS", "LCROCH", "RCROCH", "INT", "ID", 
                  "WS" ]

    grammarFileName = "gram.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


