import { Settings, MessageCircle, HelpCircle, PhoneCall } from 'lucide-react';
import { Button } from './ui/button';

export function Header() {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm px-6 py-4 flex items-center justify-between">
      <h1 className="bg-gradient-to-r from-primary via-purple-500 to-accent bg-clip-text text-transparent">
        赋范空间公开体验课
      </h1>
      
      <div className="flex items-center gap-3">
        <Button className="bg-gradient-to-r from-accent to-cyan-500 text-accent-foreground hover:shadow-[0_0_20px_rgba(0,217,255,0.5)] transition-all duration-300">
          <PhoneCall className="h-4 w-4 mr-2" />
          点击咨询课程优惠
        </Button>
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-accent transition-colors">
          <Settings className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-accent transition-colors">
          <MessageCircle className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-accent transition-colors">
          <HelpCircle className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}
