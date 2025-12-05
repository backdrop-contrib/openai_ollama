<?php

/**
 * @file
 * Ollama adapter for accessing locally-hosted AI models.
 *
 * Ollama provides OpenAI-compatible API endpoints, making integration seamless.
 * This adapter connects to a local or remote Ollama server.
 *
 * @see https://docs.ollama.com/api/openai-compatibility
 */

// Prefer Composer Manager's autoloader when available; fall back to module vendor.
$__openai_sdk_source =& backdrop_static('openai_sdk_source');
if (module_exists('composer_manager')) {
  if (function_exists('composer_manager_register_autoloader')) {
    composer_manager_register_autoloader();
  }
  $__openai_sdk_source = 'composer_manager';
}
else {
  $autoload = BACKDROP_ROOT . '/' . backdrop_get_path('module', 'openai') . '/vendor/autoload.php';
  if (file_exists($autoload)) {
    require_once $autoload;
    $__openai_sdk_source = 'module_vendor';
  }
  else {
    $__openai_sdk_source = $__openai_sdk_source ?: 'unknown';
  }
}

use OpenAI\Client as OpenAIClient;
use Symfony\Component\HttpFoundation\StreamedResponse;

class OllamaAdapter implements AIClientInterface {

  /** @var OpenAIClient */
  protected $client;

  /** @var string */
  protected $baseUrl;

  /**
   * Constructor.
   *
   * @param string $apiKey
   *   Not used for Ollama, but required by interface. Pass any string.
   */
  public function __construct($apiKey) {
    // Get base URL from central OpenAI providers mapping first, then fall back
    // to module config and finally the localhost default.
    $providers_root = config_get('openai.settings', 'providers') ?: [];
    if (!empty($providers_root['ollama']['ollama_base_url'])) {
      $this->baseUrl = $providers_root['ollama']['ollama_base_url'];
    }
    else {
      $this->baseUrl = config_get('openai_ollama.settings', 'base_url') ?: 'http://localhost:11434';
    }

    // Ensure the URL ends with /v1
    $base_uri = rtrim($this->baseUrl, '/') . '/v1';

    // Ollama doesn't require an API key, but the SDK requires one
    // Use a dummy key that won't be validated
    $dummyKey = 'ollama-local-key';

    $this->client = \OpenAI::factory()
      ->withApiKey($dummyKey)
      ->withBaseUri($base_uri)
      ->make();
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    $models = [];
    try {
      $response = $this->client->models()->list();

      foreach ($response->data as $model) {
        $id = $model->id;
        $models[$id] = $id;
      }

      if (!empty($models)) {
        asort($models);
      }
    }
    catch (\Exception $e) {
      watchdog('openai_ollama', 'Failed to fetch models: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
    }

    return $models;
  }

  /**
   * Fetch detailed model information from Ollama's native API.
   *
   * Ollama's /api/tags endpoint provides more metadata than the OpenAI-compatible
   * /v1/models endpoint, including model details that can help identify capabilities.
   *
   * @return array
   *   Array of model data with details property.
   */
  protected function fetchDetailedModels(): array {
    static $cached_models = NULL;

    if ($cached_models !== NULL) {
      return $cached_models;
    }

    $cache_key = 'ollama_detailed_models';
    $cached = cache_get($cache_key);
    if ($cached && !empty($cached->data)) {
      $cached_models = $cached->data;
      return $cached_models;
    }

    // Try Ollama's native API for more detailed model info
    $base_url = rtrim($this->baseUrl, '/');
    $native_url = $base_url . '/api/tags';

    try {
      $ch = curl_init($native_url);
      curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => TRUE,
        CURLOPT_TIMEOUT => 5,
      ]);
      $response = curl_exec($ch);
      $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
      curl_close($ch);

      if ($http_code == 200) {
        $data = json_decode($response, TRUE);
        if (isset($data['models']) && is_array($data['models'])) {
          $models = [];
          foreach ($data['models'] as $model) {
            if (isset($model['name'])) {
              $models[$model['name']] = $model;
            }
          }
          cache_set($cache_key, $models, 'cache', time() + 300); // Cache for 5 minutes
          $cached_models = $models;
          return $models;
        }
      }
    }
    catch (\Exception $e) {
      watchdog('openai_ollama', 'Failed to fetch detailed Ollama models: @error', ['@error' => $e->getMessage()], WATCHDOG_DEBUG);
    }

    $cached_models = [];
    return [];
  }

  /**
   * Get models by capability.
   *
   * NOTE: Ollama's API doesn't provide reliable capability metadata.
   * We use best-effort pattern matching and the native /api/tags endpoint.
   * Site admins can override via hook_openai_model_capabilities_alter().
   *
   * @param string $capability
   *   The capability to filter by: 'text', 'image', 'vision', 'embeddings'.
   *
   * @return array
   *   Array of model IDs => names that support the given capability.
   */
  public function getModelsByCapability($capability): array {
    // Get all models first
    $all_models = $this->getModels();

    // Try to get detailed model info from Ollama's native API
    $detailed_models = $this->fetchDetailedModels();

    $filtered = [];

    foreach ($all_models as $id => $name) {
      $ok = FALSE;

      // Check if we have detailed model info with capability metadata
      $model_details = $detailed_models[$id] ?? NULL;

      // Ollama's native API provides 'details' with 'families' array that can indicate capabilities
      // Example: models with 'clip' family support vision
      if ($model_details && isset($model_details['details']['families'])) {
        $families = $model_details['details']['families'];

        switch ($capability) {
          case 'vision':
            // Models with 'clip' family support vision (image input)
            $ok = in_array('clip', $families, TRUE);
            break;

          case 'embeddings':
          case 'embedding':
            // Check model name patterns for embeddings since families may not always indicate this
            $ok = preg_match('/(embed|embedding|nomic-embed)/i', $id);
            break;

          case 'text':
            // Most models support text; exclude embedding-only
            $ok = !preg_match('/(embed|embedding|nomic-embed)/i', $id);
            break;

          case 'image':
            // Ollama doesn't support image OUTPUT generation via OpenAI API
            $ok = FALSE;
            break;
        }
      }
      else {
        // Fall back to pattern matching if no detailed metadata available
        switch ($capability) {
          case 'text':
            // Most Ollama models support text generation (chat/completion)
            // Exclude embedding-only models
            $ok = !preg_match('/(embed|embedding|nomic-embed)/i', $id);
            break;

          case 'image':
            // Ollama doesn't support image OUTPUT generation
            $ok = FALSE;
            break;

          case 'vision':
            // Vision models (image INPUT) - expanded patterns for common models
            // Includes: llava, bakllava, llava-llama3, llava-phi3, minicpm-v, cogvlm, qwen-vl, yi-vision, etc.
            $ok = preg_match('/(llava|bakllava|vision|minicpm[-_]?v|cogvlm|qwen.*vl|yi.*vision|moondream|llama.*vision)/i', $id);
            break;

          case 'embeddings':
          case 'embedding':
            // Embedding models
            $ok = preg_match('/(embed|embedding|nomic-embed)/i', $id);
            break;
        }
      }

      if ($ok) {
        $filtered[$id] = $name;
      }
    }

    // Allow site-specific overrides via Backdrop's alter hook system
    $provider_id = 'ollama';
    backdrop_alter('openai_model_capabilities', $filtered, $capability, $provider_id);

    return $filtered;
  }

  /**
   * Get chat/text generation models.
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getChatModels(): array {
    return $this->getModelsByCapability('text');
  }

  /**
   * Get image generation models.
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getImageModels(): array {
    return $this->getModelsByCapability('image');
  }

  /**
   * Get vision models (image input).
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getVisionModels(): array {
    return $this->getModelsByCapability('vision');
  }

  /**
   * Get embedding models.
   *
   * NOTE: Ollama cannot reliably detect which models support embeddings.
   * This method returns all models - users must know which support embeddings.
   * Use hook_openai_model_capabilities_alter() to filter if needed.
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getEmbeddingModels(): array {
    // Return ALL models since we can't reliably detect embedding capability
    return $this->getModels();
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    $params = [
      'model' => $model,
      'prompt' => $prompt,
      'temperature' => (float) $temperature,
      'max_tokens' => (int) $max_tokens,
    ];

    if ($stream_response) {
      $params['stream'] = TRUE;
      $stream = $this->client->completions()->createStreamed($params);
      return $this->handleStreamedResponse($stream);
    }

    $response = $this->client->completions()->create($params);
    return $response->choices[0]->text ?? '';
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE) {
    $params = [
      'model' => $model,
      'messages' => $messages,
      'temperature' => (float) $temperature,
      'max_tokens' => (int) $max_tokens,
    ];

    if ($stream_response) {
      $params['stream'] = TRUE;
      $stream = $this->client->chat()->createStreamed($params);
      return $this->handleStreamedResponse($stream);
    }

    $response = $this->client->chat()->create($params);
    return $response->choices[0]->message->content ?? '';
  }

  /**
   * {@inheritdoc}
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    // Ollama doesn't support image generation through OpenAI-compatible API
    throw new \Exception('Image generation is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    // Ollama doesn't support TTS through OpenAI-compatible API
    throw new \Exception('Text-to-speech is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    // Ollama doesn't support STT through OpenAI-compatible API
    throw new \Exception('Speech-to-text is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function moderation(string $input, string $model = 'omni-moderation-latest'): array {
    // Ollama doesn't have built-in moderation
    throw new \Exception('Moderation is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function embedding(string $input, string $model): array {
    try {
      $response = $this->client->embeddings()->create([
        'model' => $model,
        'input' => $input,
      ]);

      return $response->embeddings[0]->embedding ?? [];
    }
    catch (\Exception $e) {
      watchdog('openai_ollama', 'Embedding failed: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return [];
    }
  }

  /**
   * Handle streamed responses.
   *
   * @param mixed $stream
   *   The stream from the API.
   *
   * @return StreamedResponse
   *   A Symfony StreamedResponse for output.
   */
  protected function handleStreamedResponse($stream): StreamedResponse {
    return new StreamedResponse(function () use ($stream) {
      foreach ($stream as $chunk) {
        $text = $chunk->choices[0]->delta->content ??
                $chunk->choices[0]->text ?? '';

        if (!empty($text)) {
          echo $text;
          ob_flush();
          flush();
        }
      }
    });
  }
}
