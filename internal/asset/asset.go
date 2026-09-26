package asset

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	LlamaVersion = "b4661"
	OrtVersion   = "1.20.1"

	DefaultLayaRepo   = "jheronimus/golaya"
	DefaultQwenRepo   = "unsloth/Qwen3-1.7B-GGUF"
	DefaultQwenFile   = "Qwen3-1.7B-Q4_K_M.gguf"
)

var (
	ErrModelNotCached = errors.New("llm: model not found in local cache")
	ErrLibNotCached   = errors.New("llm: shared library not found in local cache")
)

// EnsureCacheDir returns the root cache directory (~/.cache/llm).
func EnsureCacheDir(customDir string) (string, error) {
	if customDir != "" {
		if err := os.MkdirAll(customDir, 0o755); err != nil {
			return "", err
		}
		return customDir, nil
	}

	if env := os.Getenv("LLM_CACHE_DIR"); env != "" {
		if err := os.MkdirAll(env, 0o755); err != nil {
			return "", err
		}
		return env, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	cacheDir := filepath.Join(home, ".cache", "llm")
	if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
		cacheDir = filepath.Join(xdg, "llm")
	}

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}
	return cacheDir, nil
}

// LocateOrDownloadONNX locates or fetches libonnxruntime.
func LocateOrDownloadONNX(customPath, cacheDir string, autoDownload bool) (string, error) {
	if customPath != "" {
		if _, err := os.Stat(customPath); err == nil {
			return customPath, nil
		}
		return "", fmt.Errorf("llm: specified onnx library not found: %s", customPath)
	}

	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(cacheDir, "lib", "libonnxruntime.dylib"),
		filepath.Join(cacheDir, "lib", "libonnxruntime.so"),
		filepath.Join(home, ".cache", "golaya", "lib", "libonnxruntime.dylib"),
		filepath.Join(home, ".cache", "golaya", "lib", "libonnxruntime.so"),
		"/opt/homebrew/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.dylib",
		"/usr/lib/libonnxruntime.so",
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	if !autoDownload {
		return "", fmt.Errorf("%w: libonnxruntime (run Preload or enable WithAutoDownload)", ErrLibNotCached)
	}

	libDir := filepath.Join(cacheDir, "lib")
	if err := os.MkdirAll(libDir, 0o755); err != nil {
		return "", err
	}

	return downloadPrebuiltONNX(libDir)
}

// LocateOrDownloadLlama locates or fetches libllama.
func LocateOrDownloadLlama(customPath, cacheDir string, autoDownload bool) (string, error) {
	if customPath != "" {
		if _, err := os.Stat(customPath); err == nil {
			return customPath, nil
		}
		return "", fmt.Errorf("llm: specified libllama library not found: %s", customPath)
	}

	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(cacheDir, "lib", "libllama.dylib"),
		filepath.Join(cacheDir, "lib", "libllama.so"),
		filepath.Join(home, ".cache", "gogemma", "lib", "libllama.dylib"),
		filepath.Join(home, ".cache", "gogemma", "lib", "libllama.so"),
		".lib/libllama.dylib",
		".lib/libllama.so",
		"/opt/homebrew/lib/libllama.dylib",
		"/usr/local/lib/libllama.dylib",
		"/usr/lib/libllama.so",
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	if !autoDownload {
		return "", fmt.Errorf("%w: libllama (run Preload or enable WithAutoDownload)", ErrLibNotCached)
	}

	libDir := filepath.Join(cacheDir, "lib")
	if err := os.MkdirAll(libDir, 0o755); err != nil {
		return "", err
	}

	return downloadPrebuiltLlama(libDir)
}

// LocateOrDownloadSys1Model finds or fetches Laya ONNX model files.
func LocateOrDownloadSys1Model(customDir, modelName, cacheDir string, autoDownload bool) (string, error) {
	if customDir != "" {
		if isValidModelDir(customDir) {
			return customDir, nil
		}
		return "", fmt.Errorf("llm: invalid custom sys1 model dir: %s", customDir)
	}

	if modelName == "" {
		modelName = "laya-multilingual"
	}

	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(cacheDir, "models", modelName),
		filepath.Join(home, ".cache", "golaya", "models", modelName),
	}

	for _, c := range candidates {
		if isValidModelDir(c) {
			return c, nil
		}
	}

	if !autoDownload {
		return "", fmt.Errorf("%w: sys1 model %s", ErrModelNotCached, modelName)
	}

	targetDir := filepath.Join(cacheDir, "models", modelName)
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return "", err
	}

	url := fmt.Sprintf("https://github.com/%s/releases/latest/download/%s.tar.gz", DefaultLayaRepo, modelName)
	if err := downloadAndExtractTarGz(url, targetDir); err != nil {
		return "", fmt.Errorf("failed downloading sys1 model %s: %w", modelName, err)
	}

	if !isValidModelDir(targetDir) {
		return "", fmt.Errorf("llm: downloaded sys1 model dir %s is missing model.onnx", targetDir)
	}

	return targetDir, nil
}

// LocateOrDownloadSys2Model finds or fetches Qwen GGUF model file.
func LocateOrDownloadSys2Model(customPath, cacheDir string, autoDownload bool) (string, error) {
	if customPath != "" {
		if _, err := os.Stat(customPath); err == nil {
			return customPath, nil
		}
		return "", fmt.Errorf("llm: specified sys2 model not found: %s", customPath)
	}

	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(cacheDir, "models", DefaultQwenFile),
		filepath.Join(home, ".cache", "gogemma", "models", DefaultQwenFile),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	if !autoDownload {
		return "", fmt.Errorf("%w: sys2 model %s", ErrModelNotCached, DefaultQwenFile)
	}

	modelsDir := filepath.Join(cacheDir, "models")
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return "", err
	}

	targetPath := filepath.Join(modelsDir, DefaultQwenFile)
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", DefaultQwenRepo, DefaultQwenFile)
	if err := downloadFile(url, targetPath); err != nil {
		return "", fmt.Errorf("failed downloading sys2 model from %s: %w", url, err)
	}

	return targetPath, nil
}

// LocateOrDownloadDaemon finds or fetches the llmd sidecar daemon binary.
func LocateOrDownloadDaemon(customPath, cacheDir string, autoDownload bool) (string, error) {
	if customPath != "" {
		if _, err := os.Stat(customPath); err == nil {
			return customPath, nil
		}
		return "", fmt.Errorf("llm: specified llmd binary not found: %s", customPath)
	}

	if env := os.Getenv("LLMD_PATH"); env != "" {
		if _, err := os.Stat(env); err == nil {
			return env, nil
		}
	}

	// Check local dev builds (daemon/target/release/llmd)
	cwd, _ := os.Getwd()
	dir := cwd
	for i := 0; i < 5; i++ {
		p := filepath.Join(dir, "daemon", "target", "release", "llmd")
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	// Check ~/.cache/llm/bin/llmd
	binPath := filepath.Join(cacheDir, "bin", "llmd")
	if _, err := os.Stat(binPath); err == nil {
		return binPath, nil
	}

	// Check PATH
	if p, err := exec.LookPath("llmd"); err == nil {
		return p, nil
	}

	if !autoDownload {
		return "", fmt.Errorf("llm: llmd daemon binary not found. Set LLMD_PATH, install to PATH, or enable WithAutoDownload")
	}

	binDir := filepath.Join(cacheDir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		return "", err
	}

	return downloadPrebuiltDaemon(binDir)
}

func downloadPrebuiltDaemon(binDir string) (string, error) {
	osName := runtime.GOOS
	arch := runtime.GOARCH
	destFile := filepath.Join(binDir, "llmd")
	url := fmt.Sprintf("https://github.com/jheronimus/llm/releases/latest/download/llmd-%s-%s.tar.gz", osName, arch)
	if err := downloadAndExtractSingleFile(url, "llmd", destFile); err != nil {
		return "", fmt.Errorf("failed downloading daemon binary from %s: %w", url, err)
	}
	if err := os.Chmod(destFile, 0o755); err != nil {
		return "", err
	}
	return destFile, nil
}

func isValidModelDir(dir string) bool {
	onnxPath := filepath.Join(dir, "model.onnx")
	tokPath := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(onnxPath); err == nil {
		if _, err := os.Stat(tokPath); err == nil {
			return true
		}
	}
	return false
}

func downloadFile(url, targetPath string) error {
	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download error %s: status %d", url, resp.StatusCode)
	}

	tmpPath := targetPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	if _, err := io.Copy(f, resp.Body); err != nil {
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	_ = f.Close()

	return os.Rename(tmpPath, targetPath)
}

func downloadPrebuiltONNX(destDir string) (string, error) {
	var archiveURL, libNameInTar, targetLibName string
	switch fmt.Sprintf("%s-%s", runtime.GOOS, runtime.GOARCH) {
	case "darwin-arm64":
		archiveURL = fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-osx-arm64-%s.tgz", OrtVersion, OrtVersion)
		libNameInTar = fmt.Sprintf("onnxruntime-osx-arm64-%s/lib/libonnxruntime.%s.dylib", OrtVersion, OrtVersion)
		targetLibName = "libonnxruntime.dylib"
	case "darwin-amd64":
		archiveURL = fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-osx-x86_64-%s.tgz", OrtVersion, OrtVersion)
		libNameInTar = fmt.Sprintf("onnxruntime-osx-x86_64-%s/lib/libonnxruntime.%s.dylib", OrtVersion, OrtVersion)
		targetLibName = "libonnxruntime.dylib"
	case "linux-amd64":
		archiveURL = fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-linux-x64-%s.tgz", OrtVersion, OrtVersion)
		libNameInTar = fmt.Sprintf("onnxruntime-linux-x64-%s/lib/libonnxruntime.so.%s", OrtVersion, OrtVersion)
		targetLibName = "libonnxruntime.so"
	case "linux-arm64":
		archiveURL = fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-linux-aarch64-%s.tgz", OrtVersion, OrtVersion)
		libNameInTar = fmt.Sprintf("onnxruntime-linux-aarch64-%s/lib/libonnxruntime.so.%s", OrtVersion, OrtVersion)
		targetLibName = "libonnxruntime.so"
	default:
		return "", fmt.Errorf("llm: unsupported onnx platform %s-%s", runtime.GOOS, runtime.GOARCH)
	}

	destPath := filepath.Join(destDir, targetLibName)
	if err := downloadAndExtractSingleFile(archiveURL, libNameInTar, destPath); err != nil {
		return "", fmt.Errorf("download onnxruntime: %w", err)
	}
	return destPath, nil
}

func downloadPrebuiltLlama(destDir string) (string, error) {
	libName := "libllama.so"
	if runtime.GOOS == "darwin" {
		libName = "libllama.dylib"
	}
	targetFile := filepath.Join(destDir, libName)

	var osName, arch string
	switch runtime.GOOS {
	case "darwin":
		osName = "macos"
		switch runtime.GOARCH {
		case "arm64":
			arch = "arm64"
		case "amd64":
			arch = "x64"
		default:
			return "", fmt.Errorf("llm: unsupported llama arch %s", runtime.GOARCH)
		}
	case "linux":
		osName = "ubuntu"
		switch runtime.GOARCH {
		case "arm64":
			arch = "arm64"
		case "amd64":
			arch = "x64"
		default:
			return "", fmt.Errorf("llm: unsupported llama arch %s", runtime.GOARCH)
		}
	default:
		return "", fmt.Errorf("llm: unsupported llama os %s", runtime.GOOS)
	}

	url := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-%s-%s.tar.gz",
		LlamaVersion, LlamaVersion, osName, arch)

	tmpArchive := filepath.Join(destDir, "llama_bin.tar.gz")
	defer func() { _ = os.Remove(tmpArchive) }()

	if err := downloadFile(url, tmpArchive); err != nil {
		return "", fmt.Errorf("download libllama: %w", err)
	}

	if err := extractLibrariesFromTarGz(tmpArchive, destDir); err != nil {
		return "", fmt.Errorf("extract libllama: %w", err)
	}

	if _, err := os.Stat(targetFile); err != nil {
		candidates, _ := filepath.Glob(filepath.Join(destDir, "libllama*"))
		for _, c := range candidates {
			if strings.Contains(c, ".dylib") || strings.Contains(c, ".so") {
				_ = os.Symlink(filepath.Base(c), targetFile)
				break
			}
		}
	}

	return targetFile, nil
}

func downloadAndExtractSingleFile(url, fileInsideTar, destPath string) error {
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s failed: status %d", url, resp.StatusCode)
	}

	gzReader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer func() { _ = gzReader.Close() }()

	tarReader := tar.NewReader(gzReader)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if header.Name == fileInsideTar || strings.HasSuffix(header.Name, "/"+filepath.Base(fileInsideTar)) {
			outFile, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
			if err != nil {
				return err
			}
			if _, err := io.Copy(outFile, tarReader); err != nil {
				_ = outFile.Close()
				return err
			}
			_ = outFile.Close()
			return nil
		}
	}

	return fmt.Errorf("file %s not found in archive", fileInsideTar)
}

func downloadAndExtractTarGz(url, destDir string) error {
	client := &http.Client{Timeout: 15 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s failed: status %d", url, resp.StatusCode)
	}

	gzReader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer func() { _ = gzReader.Close() }()

	tarReader := tar.NewReader(gzReader)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		cleanPath := filepath.Clean(header.Name)
		if strings.HasPrefix(cleanPath, "..") {
			continue
		}
		target := filepath.Join(destDir, cleanPath)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, header.FileInfo().Mode())
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tarReader); err != nil {
				_ = f.Close()
				return err
			}
			_ = f.Close()
		}
	}
	return nil
}

func extractLibrariesFromTarGz(archivePath, destDir string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	gzReader, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer func() { _ = gzReader.Close() }()

	tarReader := tar.NewReader(gzReader)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		baseName := filepath.Base(header.Name)
		isLib := strings.Contains(baseName, ".dylib") || strings.Contains(baseName, ".so")
		if !isLib {
			continue
		}

		destPath := filepath.Join(destDir, baseName)
		switch header.Typeflag {
		case tar.TypeSymlink:
			_ = os.Remove(destPath)
			_ = os.Symlink(header.Linkname, destPath)
		case tar.TypeReg:
			out, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tarReader); err != nil {
				_ = out.Close()
				return err
			}
			_ = out.Close()
		}
	}
	return nil
}
