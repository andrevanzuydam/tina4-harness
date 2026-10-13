import unittest
from harness.ollama import Channels

class ChannelTests(unittest.TestCase):
    def test_arbitrary_token_boundaries(self):
        for text, initial in [('<think>reason</think>{"ok":true}', False),
                              ('reason</think>{"ok":true}', True),
                              ('<|channel>thought\nreason<channel|>{"ok":true}', False)]:
            for size in range(1, len(text)+1):
                parts=[]
                parser=Channels(lambda text, thinking: parts.append((text,thinking)), initial)
                for start in range(0,len(text),size):parser.feed(text[start:start+size])
                parser.feed('', final=True)
                self.assertEqual(''.join(t for t,k in parts if k),'reason')
                self.assertEqual(''.join(t for t,k in parts if not k),'{"ok":true}')

    def test_plain_answer_and_partial_marker(self):
        parts=[];parser=Channels(lambda text, thinking:parts.append(text))
        parser.feed('a < b');parser.feed('',final=True)
        self.assertEqual(''.join(parts),'a < b')

if __name__=='__main__':unittest.main()
