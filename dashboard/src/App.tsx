import { useState, useEffect } from 'react';
import { Activity, Camera, AlertTriangle, Users, Crosshair, MapPin, Eye, Clock, Menu } from 'lucide-react';
import './App.css'; // cleared
import './index.css';

// Mock Event Types
type EventType = 'intrusion' | 'loitering' | 'line-crossing' | 'crowd';

interface VisionEvent {
  id: string;
  type: EventType;
  camera_id: string;
  timestamp: string;
  track_id: number;
  class: string;
  zone?: string;
  details: any;
}

const INITIAL_EVENTS: VisionEvent[] = [
  {
    id: '1',
    type: 'loitering',
    camera_id: 'cam0',
    timestamp: new Date(Date.now() - 1000 * 5).toLocaleTimeString(),
    track_id: 2,
    class: 'person',
    zone: 'restricted_area_A',
    details: { dwell_seconds: 6.0 }
  },
  {
    id: '2',
    type: 'line-crossing',
    camera_id: 'cam1',
    timestamp: new Date(Date.now() - 1000 * 45).toLocaleTimeString(),
    track_id: 15,
    class: 'car',
    zone: 'main_gate',
    details: { direction: 'inbound' }
  }
];

function App() {
  const [events, setEvents] = useState<VisionEvent[]>(INITIAL_EVENTS);
  const [activeDetections, setActiveDetections] = useState(24);
  
  // Simulate incoming events
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.7) {
        const types: EventType[] = ['intrusion', 'loitering', 'line-crossing'];
        const randomType = types[Math.floor(Math.random() * types.length)];
        
        const newEvent: VisionEvent = {
          id: Math.random().toString(36).substring(7),
          type: randomType,
          camera_id: `cam${Math.floor(Math.random() * 3)}`,
          timestamp: new Date().toLocaleTimeString(),
          track_id: Math.floor(Math.random() * 100),
          class: Math.random() > 0.5 ? 'person' : 'vehicle',
          zone: 'zone_' + Math.floor(Math.random() * 5),
          details: randomType === 'loitering' ? { dwell_seconds: (Math.random() * 10 + 5).toFixed(1) } : {}
        };
        
        setEvents(prev => [newEvent, ...prev].slice(0, 50));
        
        // Slightly jitter active detections
        setActiveDetections(prev => Math.max(0, prev + (Math.random() > 0.5 ? 1 : -1)));
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const getEventIcon = (type: EventType) => {
    switch (type) {
      case 'intrusion': return <AlertTriangle size={18} />;
      case 'loitering': return <Clock size={18} />;
      case 'line-crossing': return <Crosshair size={18} />;
      case 'crowd': return <Users size={18} />;
      default: return <Activity size={18} />;
    }
  };

  const getEventColorClass = (type: EventType) => {
    switch (type) {
      case 'intrusion': return 'icon-intrusion';
      case 'loitering': return 'icon-loitering';
      case 'line-crossing': return 'icon-line';
      default: return 'icon-line';
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="glass-header">
        <h1>
          <Camera size={24} className="highlight" />
          <span>VisionPipe <span className="highlight">Dashboard</span></span>
        </h1>
        <nav className="nav-links">
          <a href="#" className="nav-link active">Live View</a>
          <a href="#" className="nav-link">Analytics</a>
          <a href="#" className="nav-link">Zone Config</a>
          <a href="#" className="nav-link"><Menu size={20} /></a>
        </nav>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="dashboard-grid">
          
          {/* Left/Main Column */}
          <div className="video-section">
            
            {/* KPI Stats Row */}
            <div className="stats-row">
              <div className="stat-card glass-panel">
                <div className="stat-header">
                  <span>Active Cameras</span>
                  <Camera size={16} />
                </div>
                <div className="stat-value">3</div>
                <div className="stat-trend trend-down">
                  <span>All streams optimal</span>
                </div>
              </div>
              
              <div className="stat-card glass-panel">
                <div className="stat-header">
                  <span>Active Tracks</span>
                  <Users size={16} />
                </div>
                <div className="stat-value">{activeDetections}</div>
                <div className="stat-trend trend-warning">
                  <span>Across all zones</span>
                </div>
              </div>

              <div className="stat-card glass-panel">
                <div className="stat-header">
                  <span>Pipeline Latency</span>
                  <Activity size={16} />
                </div>
                <div className="stat-value">22.4 <span style={{fontSize: '1rem', color: 'var(--text-tertiary)'}}>FPS</span></div>
                <div className="stat-trend trend-down">
                  <span>ONNX CPU backend</span>
                </div>
              </div>

              <div className="stat-card glass-panel">
                <div className="stat-header">
                  <span>Dropped Frames</span>
                  <AlertTriangle size={16} />
                </div>
                <div className="stat-value">0</div>
                <div className="stat-trend trend-down">
                  <span>0% in last hour</span>
                </div>
              </div>
            </div>

            {/* Video Player */}
            <div className="video-container">
              <img 
                src="/camera-feed.jpg" 
                alt="Security Camera View" 
                className="video-placeholder"
              />
              <div className="video-overlay">
                <div className="badge-live">
                  <div className="live-indicator"></div>
                  LIVE • CAM0
                </div>
              </div>
            </div>

          </div>

          {/* Right Sidebar - Event Feed */}
          <aside className="events-sidebar glass-panel">
            <div className="events-header">
              <div className="events-title">
                <Activity size={18} className="highlight" />
                Live Event Stream
              </div>
              <span style={{fontSize: '0.75rem', color: 'var(--text-tertiary)'}}>{events.length} events</span>
            </div>
            
            <div className="events-list">
              {events.map((evt) => (
                <div key={evt.id} className="event-item">
                  <div className={`event-icon ${getEventColorClass(evt.type)}`}>
                    {getEventIcon(evt.type)}
                  </div>
                  <div className="event-content">
                    <div className="event-top">
                      <span className="event-type">{evt.type.replace('-', ' ')}</span>
                      <span className="event-time">{evt.timestamp}</span>
                    </div>
                    <div className="event-details">
                      <span style={{display: 'flex', alignItems: 'center', gap: '0.25rem'}}>
                        <Eye size={12} /> {evt.class} (ID: {evt.track_id})
                      </span>
                      <span style={{display: 'flex', alignItems: 'center', gap: '0.25rem'}}>
                        <MapPin size={12} /> {evt.zone}
                      </span>
                    </div>
                    {evt.details.dwell_seconds && (
                      <div style={{fontSize: '0.75rem', color: 'var(--status-warning)', marginTop: '0.25rem'}}>
                        Dwell time: {evt.details.dwell_seconds}s
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </aside>

        </div>
      </main>
    </div>
  );
}

export default App;
