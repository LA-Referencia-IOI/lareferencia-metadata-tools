from importlib.metadata import distribution
import unittest


class EntryPointsTest(unittest.TestCase):
    def test_console_commands_resolve(self):
        entry_points = distribution('lareferencia-metadata-tools').entry_points
        for entry in entry_points:
            if entry.group == 'console_scripts':
                with self.subTest(command=entry.name):
                    self.assertTrue(callable(entry.load()))
