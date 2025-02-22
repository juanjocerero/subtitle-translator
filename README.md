# Subtitle Translator

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/juanjocerero/subtitle-translator/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/juanjocerero/subtitle-translator/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/juanjocerero/subtitle-translator/blob/main/CONTRIBUTING.md)

A powerful subtitle translation tool using NLLB (No Language Left Behind) for high-quality translations with context awareness and quality control.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/juanjocerero/subtitle-translator.git
cd subtitle-translator

# Install dependencies
pip install -r requirements.txt

# Make the script executable
chmod +x translate-subs

# Optional: Install globally
sudo cp translate-subs /usr/local/bin/
sudo chmod +x /usr/local/bin/translate-subs

# Translate a subtitle file
translate-subs movie.srt
```

## Features

### Core Translation
- Uses Facebook's NLLB-200 model for high-quality translations
- Supports 200+ languages with extended mode
- Automatic language detection for source files
- Context-aware translation (considers surrounding subtitles)
- Preserves subtitle formatting and timing
- GPU acceleration with CUDA when available

### Quality Control
- Quality analysis for each translation:
  - Length ratio verification
  - Punctuation consistency check
  - Number preservation
  - Proper name preservation
  - Context coherence analysis
- Quality scoring system (0-100)
- Detailed quality reports in logs

### Smart Processing
- Batch processing for efficiency
- Automatic memory optimization
- Progress tracking with ETA
- Checkpoint system for resuming interrupted translations
- UTF-8 encoding handling
- Format tag preservation

### Language Support
- Basic mode: 6 main languages
- Extended mode: 200+ languages
- Automatic language detection
- Language code mapping and validation
- Support for regional variants
- Script variants (e.g., simplified/traditional Chinese)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/subtitle-translator.git
cd subtitle-translator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Command Line Wrapper

The project includes a convenient command-line wrapper `translate-subs` that makes it easier to use the translator.

### Installation

1. **Local Installation**:
```bash
# Give execution permissions
chmod +x translate-subs

# Create symbolic link (optional)
ln -s "$(pwd)/translate-subs" ~/bin/translate-subs
```

2. **Global Installation**:
```bash
# Create installation directories
sudo mkdir -p /usr/local/lib/subtitle-translator
sudo mkdir -p /usr/local/bin

# Copy project files
sudo cp -r src/translator.py src/languages.py requirements.txt /usr/local/lib/subtitle-translator/
sudo cp translate-subs /usr/local/bin/
sudo chmod +x /usr/local/bin/translate-subs
```

The script will automatically create and manage:
- Virtual environment in `~/.local/share/subtitle-translator/venv`
- Configuration in `~/.config/subtitle-translator`
- Log files in `~/.config/subtitle-translator`

### Usage

```bash
# Show help
translate-subs --help

# Basic translation (language autodetection)
translate-subs video.srt

# Specify target language
translate-subs -t french video.srt

# Use extended languages
translate-subs -e -s japanese -t chinese video.srt

# List supported languages
translate-subs --list-languages

# List all available languages
translate-subs --list-languages --extended

# Clean temporary files
translate-subs --clean
```

### Options

| Short Option | Long Option | Description |
|--------------|-------------|-------------|
| -h | --help | Show help message |
| -l | --list-languages | List supported languages |
| -e | --extended | Enable all languages |
| -s | --source LANG | Source language |
| -t | --target LANG | Target language |
| -c | --config FILE | Configuration file |
| | --clean | Remove temporary files |
| | --version | Show version |

### Features

- **Environment Management**:
  - Automatic virtual environment creation
  - Dependency installation
  - System requirements verification

- **Enhanced Interface**:
  - Short and long arguments
  - Colored messages
  - Progress indicators
  - Detailed error handling

- **Additional Functions**:
  - Temporary file cleanup
  - Input file verification
  - Filename autocompletion
  - Detailed logging

### Usage Examples

1. **Simple Translation**:
   ```bash
   # Translate to Spanish (default)
   translate-subs movie.srt
   
   # Specify output file
   translate-subs movie.srt movie_es.srt
   ```

2. **Specific Languages**:
   ```bash
   # Translate from English to French
   translate-subs -s english -t french movie.srt
   
   # Use language autodetection
   translate-subs -t german movie.srt
   ```

3. **Extended Languages**:
   ```bash
   # Use additional languages
   translate-subs -e -s japanese -t "chinese (simplified)" movie.srt
   
   # View all available languages
   translate-subs -e -l
   ```

4. **Advanced Configuration**:
   ```bash
   # Use configuration file
   translate-subs -c config.json movie.srt
   
   # Clean temporary files
   translate-subs --clean
   ```

### Maintenance

- Logs are saved in `translation.log`
- Checkpoints are saved in `translation_checkpoint.json`
- Use `translate-subs --clean` to remove temporary files
- Virtual environment is created in `.venv/`

## Configuration

The translator can be configured using a JSON configuration file. Here are all available options:

```json
{
  "max_length": 512,        // Maximum token length for translation
  "context_window": 3,      // Number of surrounding subtitles to consider
  "quality_threshold": 70,  // Minimum quality score (0-100)
  "save_interval": 50,      // Number of subtitles between checkpoints
  "retry_attempts": 3,      // Number of retries for failed translations
  "batch_size": null       // Batch size (null for auto-calculation)
}
```

### Configuration Details

#### Translation Parameters
- `max_length`: Maximum number of tokens for translation (default: 512)
  - Increase for longer subtitles
  - Decrease to save memory
- `context_window`: Number of surrounding subtitles for context (default: 3)
  - Higher values provide better context
  - Lower values improve processing speed

#### Quality Control
- `quality_threshold`: Minimum acceptable quality score (default: 70)
  - Scores below this trigger warnings
  - Based on multiple factors:
    * Length ratio (30%)
    * Punctuation (20%)
    * Numbers (20%)
    * Names (20%)
    * Context (10%)

#### Processing Options
- `save_interval`: Subtitles between checkpoints (default: 50)
  - Lower values for more frequent saves
  - Higher values for better performance
- `retry_attempts`: Failed translation retries (default: 3)
  - Includes exponential backoff
- `batch_size`: Translation batch size
  - null for automatic optimization
  - Manual setting for specific requirements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
