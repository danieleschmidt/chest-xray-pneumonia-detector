// MongoDB initialization script for Chest X-Ray Pneumonia Detector

// Switch to the application database
db = db.getSiblingDB('cxr_metadata');

// Create collections with validation schemas

// Image metadata collection
db.createCollection('image_metadata', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['image_id', 'filename', 'upload_timestamp'],
      properties: {
        image_id: {
          bsonType: 'string',
          description: 'Unique identifier for the image'
        },
        filename: {
          bsonType: 'string',
          description: 'Original filename of the uploaded image'
        },
        file_size: {
          bsonType: 'number',
          description: 'File size in bytes'
        },
        image_dimensions: {
          bsonType: 'object',
          properties: {
            width: { bsonType: 'number' },
            height: { bsonType: 'number' },
            channels: { bsonType: 'number' }
          }
        },
        upload_timestamp: {
          bsonType: 'date',
          description: 'When the image was uploaded'
        },
        processing_status: {
          bsonType: 'string',
          enum: ['pending', 'processing', 'completed', 'failed'],
          description: 'Current processing status'
        },
        anonymization_status: {
          bsonType: 'string',
          enum: ['pending', 'completed', 'not_required'],
          description: 'HIPAA anonymization status'
        },
        tags: {
          bsonType: 'array',
          items: { bsonType: 'string' },
          description: 'User-defined tags for the image'
        }
      }
    }
  }
});

// Model performance metrics collection
db.createCollection('model_metrics', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['model_version', 'metric_name', 'metric_value', 'timestamp'],
      properties: {
        model_version: {
          bsonType: 'string',
          description: 'Version of the model'
        },
        metric_name: {
          bsonType: 'string',
          description: 'Name of the performance metric'
        },
        metric_value: {
          bsonType: 'number',
          description: 'Value of the metric'
        },
        dataset_info: {
          bsonType: 'object',
          properties: {
            dataset_name: { bsonType: 'string' },
            dataset_size: { bsonType: 'number' },
            evaluation_date: { bsonType: 'date' }
          }
        },
        timestamp: {
          bsonType: 'date',
          description: 'When the metric was recorded'
        }
      }
    }
  }
});

// Data pipeline logs collection
db.createCollection('pipeline_logs', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['pipeline_id', 'step', 'status', 'timestamp'],
      properties: {
        pipeline_id: {
          bsonType: 'string',
          description: 'Unique identifier for the pipeline run'
        },
        step: {
          bsonType: 'string',
          description: 'Pipeline step name'
        },
        status: {
          bsonType: 'string',
          enum: ['started', 'completed', 'failed', 'skipped'],
          description: 'Status of the pipeline step'
        },
        duration_ms: {
          bsonType: 'number',
          description: 'Duration of the step in milliseconds'
        },
        input_count: {
          bsonType: 'number',
          description: 'Number of input items processed'
        },
        output_count: {
          bsonType: 'number',
          description: 'Number of output items produced'
        },
        error_message: {
          bsonType: 'string',
          description: 'Error message if step failed'
        },
        timestamp: {
          bsonType: 'date',
          description: 'When the step was executed'
        }
      }
    }
  }
});

// User sessions collection for API usage tracking
db.createCollection('user_sessions', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['session_id', 'user_id', 'start_time'],
      properties: {
        session_id: {
          bsonType: 'string',
          description: 'Unique session identifier'
        },
        user_id: {
          bsonType: 'string',
          description: 'User identifier'
        },
        start_time: {
          bsonType: 'date',
          description: 'Session start time'
        },
        end_time: {
          bsonType: 'date',
          description: 'Session end time'
        },
        actions_count: {
          bsonType: 'number',
          description: 'Number of actions performed in session'
        },
        ip_address: {
          bsonType: 'string',
          description: 'Client IP address'
        },
        user_agent: {
          bsonType: 'string',
          description: 'Client user agent'
        }
      }
    }
  }
});

// Create indexes for performance
db.image_metadata.createIndex({ 'image_id': 1 }, { unique: true });
db.image_metadata.createIndex({ 'upload_timestamp': -1 });
db.image_metadata.createIndex({ 'processing_status': 1 });

db.model_metrics.createIndex({ 'model_version': 1, 'timestamp': -1 });
db.model_metrics.createIndex({ 'metric_name': 1, 'timestamp': -1 });

db.pipeline_logs.createIndex({ 'pipeline_id': 1, 'timestamp': 1 });
db.pipeline_logs.createIndex({ 'step': 1, 'status': 1 });
db.pipeline_logs.createIndex({ 'timestamp': -1 });

db.user_sessions.createIndex({ 'user_id': 1, 'start_time': -1 });
db.user_sessions.createIndex({ 'session_id': 1 }, { unique: true });

// Insert sample data for development
if (db.image_metadata.countDocuments() === 0) {
  db.image_metadata.insertMany([
    {
      image_id: 'sample_001',
      filename: 'chest_xray_normal_001.jpg',
      file_size: 145000,
      image_dimensions: { width: 1024, height: 1024, channels: 1 },
      upload_timestamp: new Date(),
      processing_status: 'completed',
      anonymization_status: 'completed',
      tags: ['normal', 'chest', 'adult']
    },
    {
      image_id: 'sample_002',
      filename: 'chest_xray_pneumonia_001.jpg',
      file_size: 167000,
      image_dimensions: { width: 1024, height: 1024, channels: 1 },
      upload_timestamp: new Date(),
      processing_status: 'completed',
      anonymization_status: 'completed',
      tags: ['pneumonia', 'chest', 'adult']
    }
  ]);
}

print('MongoDB initialization completed for cxr_metadata database');